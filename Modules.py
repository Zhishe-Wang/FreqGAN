from args import args
from torch.autograd import Variable
import torch
import os
from utils import make_floor
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import pywt
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-5


class Multi_feature_Fusion_Module(nn.Module):
    def __init__(self, channel, is_last=False):
        super(Multi_feature_Fusion_Module, self).__init__()

        self.is_last = is_last

        self.SE_Block = SE_Block(channel, is_dis=False)

        self.Up = nn.Upsample(scale_factor=2)

        self.Conv1_1 = ConvLayer(channel * 2, channel, 3, 1, True)
        self.Conv1_2 = ConvLayer(channel, channel, 3, 1, True)
        self.Conv4_2 = ConvLayer(16, 1, 3, 1, True)

    def forward(self, Fused_Image, FeatureMap_Ir, Feats_Ir, FeatureMap_Vis, Feats_Vis):
        add_Ir = self.Conv1_1(FeatureMap_Ir) + Feats_Ir
        add_Vis = self.Conv1_1(FeatureMap_Vis) + Feats_Vis
        Fused_Image = self.Conv1_1(Fused_Image)
        w = self.SE_Block(Fused_Image)
        add_all = w * add_Ir + (1 - w) * add_Vis
        out = self.Conv1_2(self.Up(add_all))

        if self.is_last is True:
            out = self.Conv4_2(self.Up(add_all))

        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.PReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Prelu is True:
            out_F = self.PReLU(out)
            return out_F


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(Conv, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is True:
            out = F.leaky_relu(out, inplace=True)
        return out


class ConvLayer_Dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_Leakyrelu=True):
        super(ConvLayer_Dis, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_Leakyrelu = use_Leakyrelu
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv2d(x)
        if self.use_Leakyrelu is True:
            out = self.LeakyReLU(out)
        return out


class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, shape, x):
        x = self.up(x)

        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, channel, is_dis=False):
        super(SE_Block, self).__init__()

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.is_dis = is_dis

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.AvgPool(x).view(b, c)
        y = self.FC(y).view(b, c, 1, 1)
        out = x * y

        if self.is_dis is True:
            out = y

        return out


class DWT_cat(nn.Module):
    def __init__(self):
        super(DWT_cat, self).__init__()
        self.dwt = DWT_2D(wavename='haar')  # return LL,LH,HL,HH
        self.Conv_1 = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        self.Conv_2 = nn.Conv2d(3, 1, kernel_size=1, stride=1)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        out = torch.cat((LL, LH, HL, HH), dim=1)
        return out


class DWT(nn.Module):
    def __init__(self, in_channels):
        super(DWT, self).__init__()
        self.dwt = DWT_2D(wavename='haar')  # return LL,LH,HL,HH
        self.Conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        LL = self.Conv(LL)
        out_WS = LH + HL + HH
        return out_WS, LL

class DWT_2D(nn.Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc:    (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):  # 获取矩阵
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}
                                 \mathcal{H}
        :return: self.matrix_low = \mathcal{L}
                 self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                     0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                     0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, shape, x):
        x = self.up(x)

        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x


def _generate_fusion_image(G_model, ir_img, vis_img):
    f = G_model(ir_img, vis_img)
    return f


def generate(model, ir_path, vis_path, result, index, mode):
    result = "results"
    out = utils.get_image(vis_path, height=None, width=None)

    ir_img = utils.get_test_images(ir_path, mode=mode)
    vis_img = utils.get_test_images(vis_path, mode=mode)
    ir_img = ir_img.cuda()
    vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)

    img_fusion = _generate_fusion_image(model, ir_img, vis_img)
    img_fusion = (img_fusion / 2 + 0.5) * 255

    img_fusion = img_fusion.squeeze()
    img_fusion = img_fusion

    if args.cuda:
        img = img_fusion.cpu().clamp(0, 255).data.numpy()
    else:
        img = img_fusion.clamp(0, 255).data[0].numpy()

    result_path = make_floor(os.getcwd(), result)

    if index < 100:
        f_filenames = "1" + str(index) + '.png'
        output_path = result_path + '/' + f_filenames
        utils.save_images(output_path, img, out)
    elif index < 10:
        f_filenames = "100" + str(index) + '.png'
        output_path = result_path + '/' + f_filenames
        utils.save_images(output_path, img, out)
    else:
        f_filenames = str(index) + '.png'
        output_path = result_path + '/' + f_filenames
        utils.save_images(output_path, img, out)
    print(output_path)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.PReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Prelu is True:
            out_F = self.PReLU(out)
            return out_F
