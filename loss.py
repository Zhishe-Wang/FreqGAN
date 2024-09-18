from torch import nn
import torch
import torch.nn.functional as F
from math import exp
from args import args
from Modules import DWT_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class g_content_loss(nn.Module):
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.SSIM_loss = L_SSIM()
        self.Grad_loss = L_Grad()
        self.Intensity_loss = L_Intensity()

    def forward(self, img_ir, img_vi, img_fusion):
        SSIM_loss = self.SSIM_loss(img_ir, img_vi, img_fusion)
        Grad_loss = self.Grad_loss(img_ir, img_vi, img_fusion)
        Intensity_loss = self.Intensity_loss(img_ir, img_vi, img_fusion)
        total_loss = args.weight_SSIM * (
                1 - SSIM_loss) + args.weight_Grad * Grad_loss + args.weight_Intensity * Intensity_loss
        return total_loss, SSIM_loss, Intensity_loss, Grad_loss


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()
        self.DWT = DWT_2D(wavename='haar')

    def forward(self, image_A, image_B, image_fused):
        r = args.r
        LL_ir, LH_ir, HL_ir, HH_ir = self.DWT(image_A)
        LL_vis, LH_vis, HL_vis, HH_vis = self.DWT(image_B)
        LL_fused, LH_fused, HL_fused, HH_fused = self.DWT(image_fused)
        ssim_ir = (r * r) * ssim(LL_ir, LL_fused) + (r * (1 - r)) * ssim(LH_ir, LH_fused) \
                  + (r * (1 - r)) * ssim(HL_ir, HL_fused) + ((1 - r) * (1 - r)) * ssim(HH_ir, HH_fused)
        ssim_vis = (r * r) * ssim(LL_vis, LL_fused) + (r * (1 - r)) * ssim(LH_vis, LH_fused) \
                   + (r * (1 - r)) * ssim(HL_vis, HL_fused) + ((1 - r) * (1 - r)) * ssim(HH_vis, HH_fused)

        Loss_SSIM = ssim_ir + ssim_vis
        return Loss_SSIM


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()
        self.DWT = DWT_2D(wavename='haar')

    def forward(self, image_A, image_B, image_fused):
        r = args.r

        LL_ir, LH_ir, HL_ir, HH_ir = self.DWT(image_A)
        LL_vis, LH_vis, HL_vis, HH_vis = self.DWT(image_B)
        LL_fused, LH_fused, HL_fused, HH_fused = self.DWT(image_fused)

        gradient_LL = torch.max(self.sobelconv(LL_ir), self.sobelconv(LL_vis))
        Loss_gradient_LL = F.l1_loss(LL_fused, gradient_LL)
        gradient_LH = torch.max(self.sobelconv(LH_ir), self.sobelconv(LH_vis))
        Loss_gradient_LH = F.l1_loss(LH_fused, gradient_LH)
        gradient_HL = torch.max(self.sobelconv(HL_ir), self.sobelconv(HL_vis))
        Loss_gradient_HL = F.l1_loss(HL_fused, gradient_HL)
        gradient_HH = torch.max(self.sobelconv(HH_ir), self.sobelconv(HH_vis))
        Loss_gradient_HH = F.l1_loss(HH_fused, gradient_HH)

        Loss_gradient = (r * r) * Loss_gradient_LL + (r * (1 - r)) * Loss_gradient_LH \
                        + (r * (1 - r)) * Loss_gradient_HL + ((1 - r) * (1 - r)) * Loss_gradient_HH
        return Loss_gradient


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()
        self.DWT = DWT_2D(wavename='haar')

    def forward(self, image_A, image_B, image_fused):
        r = args.r

        LL_ir, LH_ir, HL_ir, HH_ir = self.DWT(image_A)
        LL_vis, LH_vis, HL_vis, HH_vis = self.DWT(image_B)
        LL_fused, LH_fused, HL_fused, HH_fused = self.DWT(image_fused)

        intensity_joint_LL = torch.max(LL_ir, LL_vis)
        Loss_intensity_LL = F.l1_loss(LL_fused, intensity_joint_LL)
        intensity_joint_LH = torch.max(LH_ir, LH_vis)
        Loss_intensity_LH = F.l1_loss(LH_fused, intensity_joint_LH)
        intensity_joint_HL = torch.max(HL_ir, HL_vis)
        Loss_intensity_HL = F.l1_loss(HL_fused, intensity_joint_HL)
        intensity_joint_HH = torch.max(HH_ir, HH_vis)
        Loss_intensity_HH = F.l1_loss(HH_fused, intensity_joint_HH)

        Loss_intensity = (r * r) * Loss_intensity_LL + (r * (1 - r)) * Loss_intensity_LH \
                         + (r * (1 - r)) * Loss_intensity_HL + ((1 - r) * (1 - r)) * Loss_intensity_HH

        return Loss_intensity


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class grad(nn.Module):
    def __init__(self, channels=1):
        super(grad, self).__init__()
        laplacian_kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]).float()

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.laplacian_filter(x) ** 2


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(
        0)  # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret


def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(256, 256), kernel_size=(1, 1))
    return res


# 方差计算
def std(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1


def sum(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    win1 = torch.ones_like(window)
    res = F.conv2d(img, win1, padding=padd, groups=channel)
    return res


def final_ssim(img_ir, img_vis, img_fuse):
    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()


def final_mse(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    w_vi = torch.where(img_ir <= m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    res = map1 * mse_ir + map2 * mse_vi
    res = res * w_vi
    return res.mean()
