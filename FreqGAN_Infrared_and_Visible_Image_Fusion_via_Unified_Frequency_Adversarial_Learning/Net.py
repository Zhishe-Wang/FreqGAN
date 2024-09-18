import torch
import torch.nn as nn
from Modules import DWT, SE_Block, ConvLayer, Conv, ConvLayer_Dis, Multi_feature_Fusion_Module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator_DWT(nn.Module):
    def __init__(self):
        super(Generator_DWT, self).__init__()

        # Encoder
        self.ConvBlock_1 = nn.Sequential(ConvLayer(1, 16, 3, 1, True), ConvLayer(16, 16, 3, 1, True))
        self.ConvBlock_2 = nn.Sequential(ConvLayer(16, 32, 3, 1, True), ConvLayer(32, 32, 3, 2, True))
        self.ConvBlock_3 = nn.Sequential(ConvLayer(32, 64, 3, 1, True), ConvLayer(64, 64, 3, 2, True))
        self.ConvBlock_4 = nn.Sequential(ConvLayer(64, 128, 3, 1, True), ConvLayer(128, 128, 3, 2, True))
        self.ConvBlock_5 = nn.Sequential(ConvLayer(128, 256, 3, 1, True), ConvLayer(256, 256, 3, 2, True))

        # Wavelet
        self.DWT_1 = DWT(16)
        self.DWT_2 = DWT(32)
        self.DWT_3 = DWT(64)
        self.DWT_4 = DWT(128)

        # MFM : Multi_feature Fusion Module
        self.MFM_1 = Multi_feature_Fusion_Module(128)
        self.MFM_2 = Multi_feature_Fusion_Module(64)
        self.MFM_3 = Multi_feature_Fusion_Module(32)
        self.MFM_4 = Multi_feature_Fusion_Module(16, is_last=True)

    def forward(self, Input_Ir, Input_Vis):
        # Encoder_Ir
        Ir_1 = self.ConvBlock_1(Input_Ir)
        FeatsHigh_Ir_1, LL1 = self.DWT_1(Ir_1)

        Ir_2 = self.ConvBlock_2(Ir_1)
        FeatsHigh_Ir_2, LL2 = self.DWT_2(Ir_2)

        Ir_3 = self.ConvBlock_3(Ir_2 + LL1)
        FeatsHigh_Ir_3, LL3 = self.DWT_3(Ir_3)

        Ir_4 = self.ConvBlock_4(Ir_3 + LL2)
        FeatsHigh_Ir_4, LL4 = self.DWT_4(Ir_4)

        Ir_5 = self.ConvBlock_5(Ir_4 + LL3)
        Ir_5 = Ir_5 + LL4

        # Encoder_Vis
        Vis_1 = self.ConvBlock_1(Input_Vis)
        FeatsHigh_Vis_1, LL1 = self.DWT_1(Vis_1)

        Vis_2 = self.ConvBlock_2(Vis_1)
        FeatsHigh_Vis_2, LL2 = self.DWT_2(Vis_2)

        Vis_3 = self.ConvBlock_3(Vis_2 + LL1)
        FeatsHigh_Vis_3, LL3 = self.DWT_3(Vis_3)

        Vis_4 = self.ConvBlock_4(Vis_3 + LL2)
        FeatsHigh_Vis_4, LL4 = self.DWT_4(Vis_4)

        Vis_5 = self.ConvBlock_5(Vis_4 + LL3)
        Vis_5 = Vis_5 + LL4

        # FusionLayer
        FusedImage = Ir_5 + Vis_5

        # Decoder
        Recon_1 = self.MFM_1(FusedImage, Ir_5, FeatsHigh_Ir_4, Vis_5, FeatsHigh_Vis_4)
        Recon_2 = self.MFM_2(Recon_1, Ir_4, FeatsHigh_Ir_3, Vis_4, FeatsHigh_Vis_3)
        Recon_3 = self.MFM_3(Recon_2, Ir_3, FeatsHigh_Ir_2, Vis_3, FeatsHigh_Vis_2)
        Recon_4 = self.MFM_4(Recon_3, Ir_2, FeatsHigh_Ir_1, Vis_2, FeatsHigh_Vis_1)

        output = Recon_4
        return output


class D_IR(nn.Module):
    def __init__(self):
        super(D_IR, self).__init__()

        self.Conv_1 = nn.Sequential(ConvLayer_Dis(4, 4, 3, 2, 1), ConvLayer_Dis(4, 8, 3, 1, 1))
        self.Conv_2 = nn.Sequential(ConvLayer_Dis(8, 8, 3, 2, 1), ConvLayer_Dis(8, 16, 3, 1, 1))
        self.Conv_3 = nn.Sequential(ConvLayer_Dis(16, 16, 3, 2, 1), ConvLayer_Dis(16, 32, 3, 1, 1))
        self.Conv_4 = nn.Sequential(ConvLayer_Dis(32, 32, 3, 2, 1), ConvLayer_Dis(32, 64, 3, 1, 1))

        self.SE_Block_1 = SE_Block(8, is_dis=True)
        self.SE_Block_2 = SE_Block(16, is_dis=True)
        self.SE_Block_3 = SE_Block(32, is_dis=True)
        self.SE_Block_4 = SE_Block(64, is_dis=True)

        self.ConvFC = nn.Sequential(ConvLayer_Dis(64, 64, 1, 1, 1), ConvLayer_Dis(64, 64, 1, 1, 1))

    def forward(self, x):
        x1 = self.Conv_1(x)
        w1 = self.SE_Block_1(x1)
        x1 = w1 * x1

        x2 = self.Conv_2(x1)
        w2 = self.SE_Block_2(x2)
        x2 = w2 * x2

        x3 = self.Conv_3(x2)
        w3 = self.SE_Block_3(x3)
        x3 = w3 * x3

        x4 = self.Conv_4(x3)
        w4 = self.SE_Block_4(x4)
        out = w4 * x4

        out = self.ConvFC(out)

        return out


class D_VI(nn.Module):
    def __init__(self):
        super(D_VI, self).__init__()

        self.Conv_1 = nn.Sequential(ConvLayer_Dis(4, 4, 3, 2, 1), ConvLayer_Dis(4, 8, 3, 1, 1))
        self.Conv_2 = nn.Sequential(ConvLayer_Dis(8, 8, 3, 2, 1), ConvLayer_Dis(8, 16, 3, 1, 1))
        self.Conv_3 = nn.Sequential(ConvLayer_Dis(16, 16, 3, 2, 1), ConvLayer_Dis(16, 32, 3, 1, 1))
        self.Conv_4 = nn.Sequential(ConvLayer_Dis(32, 32, 3, 2, 1), ConvLayer_Dis(32, 64, 3, 1, 1))

        self.SE_Block_1 = SE_Block(8)
        self.SE_Block_2 = SE_Block(16)
        self.SE_Block_3 = SE_Block(32)
        self.SE_Block_4 = SE_Block(64)

        self.ConvFC = nn.Sequential(ConvLayer_Dis(64, 64, 1, 1, 1), ConvLayer_Dis(64, 64, 1, 1, 1))

    def forward(self, x):
        x1 = self.Conv_1(x)
        w1 = self.SE_Block_1(x1)
        x1 = w1 * x1

        x2 = self.Conv_2(x1)
        w2 = self.SE_Block_2(x2)
        x2 = w2 * x2

        x3 = self.Conv_3(x2)
        w3 = self.SE_Block_3(x3)
        x3 = w3 * x3

        x4 = self.Conv_4(x3)
        w4 = self.SE_Block_4(x4)
        out = w4 * x4

        out = self.ConvFC(out)

        return out
