"""
UNet_BN_ECA_DecoderPyramid: BN + ECA @ E4 + Decoder Pyramid（三合一）

组合逻辑
────────
  三个模块均已单独验证有效（在无 BN 的 UNet 上）：
    UNet_BN              : mIoU 0.8677  StemIoU 0.6745（最优单模块）
    UNet_E4_ECA          : mIoU 0.8623  StemIoU 0.6662
    UNet_Decoder_Pyramid : mIoU 0.8641  StemIoU 0.6677（best run）

三者机制完全正交：
  BN              → 全局激活归一化，改善优化
  ECA @ E4        → 编码器通道注意力，聚焦茎部特征通道
  DecoderPyramid  → 解码器跨层语义融合，细节恢复更完整

架构流程
─────────
  Encoder(BN):
    c1→c2→c3→enc4(BN)→ECA→c4→enc5(BN)→c5

  Decoder(BN):
    d4=dec4(BN)[up(c5),c4] → d3=dec3(BN)[up(d4),c3]
    → d2=dec2(BN)[up(d3),c2] → d1=dec1(BN)[up(d2),c1]

  DecoderPyramid:
    [d4,d3,d2,d1] → FPN top-down → pd1

  out = Conv1×1(pd1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ECA(nn.Module):
    def __init__(self, channels, k=5):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class DecoderPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [512, 256, 128, 64]
        self.lateral  = nn.ModuleList([nn.Conv2d(c, c, 1, bias=False) for c in chs])
        self.top_down = nn.ModuleList([nn.Conv2d(chs[i], chs[i+1], 1, bias=False) for i in range(3)])
        self.smooth   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for c in chs
        ])

    def forward(self, features):
        d4, d3, d2, d1 = features
        pd4 = self.smooth[0](self.lateral[0](d4))
        pd3 = self.smooth[1](self.lateral[1](d3) +
              self.top_down[0](F.interpolate(pd4, size=d3.shape[2:], mode='bilinear', align_corners=False)))
        pd2 = self.smooth[2](self.lateral[2](d2) +
              self.top_down[1](F.interpolate(pd3, size=d2.shape[2:], mode='bilinear', align_corners=False)))
        pd1 = self.smooth[3](self.lateral[3](d1) +
              self.top_down[2](F.interpolate(pd2, size=d1.shape[2:], mode='bilinear', align_corners=False)))
        return pd1


class UNet_BN_ECA_DecoderPyramid(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        # ── Encoder（全部 BNConvBlock）────────────────────────
        self.enc1 = BNConvBlock(input_channels, 64)
        self.enc2 = BNConvBlock(64, 128)
        self.enc3 = BNConvBlock(128, 256)
        self.enc4 = BNConvBlock(256, 512)
        self.eca  = ECA(512)                     # ECA @ E4
        self.enc5 = BNConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        # ── Decoder（全部 BNConvBlock）────────────────────────
        self.up4  = UpConv(1024, 512)
        self.dec4 = BNConvBlock(1024, 512)
        self.up3  = UpConv(512, 256)
        self.dec3 = BNConvBlock(512, 256)
        self.up2  = UpConv(256, 128)
        self.dec2 = BNConvBlock(256, 128)
        self.up1  = UpConv(128, 64)
        self.dec1 = BNConvBlock(128, 64)
        # ── Decoder Pyramid ───────────────────────────────────
        self.dec_pyramid = DecoderPyramid()
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.eca(self.enc4(self.pool(c3)))    # BN + ECA @ E4
        c5 = self.enc5(self.pool(c4))
        d4 = self.dec4(torch.cat([self.up4(c5), c4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        pd1 = self.dec_pyramid([d4, d3, d2, d1])
        return self.out(pd1)
