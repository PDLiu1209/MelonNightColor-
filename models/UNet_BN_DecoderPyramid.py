"""
UNet_BN_DecoderPyramid: BN + Decoder-side FPN Top-down Pyramid

组合逻辑
────────
  UNet_BN              : mIoU 0.8677  StemIoU 0.6745
  UNet_Decoder_Pyramid : mIoU 0.8641  StemIoU 0.6677（best run）

DecoderPyramid 通过 FPN top-down 融合 d4→d3→d2→d1，让浅层解码器
获得深层解码器的语义信息，弥补 UNet 解码器各层独立、缺乏跨层融合的缺陷。

BN 使每一层的特征分布更稳定，DecoderPyramid 中的 lateral/top-down
融合后的加法操作在 BN 规范的特征空间中更稳定，期望两者协同提升。

架构流程
─────────
  Encoder(BN) → Decoder(BN) → [d4,d3,d2,d1] → DecoderPyramid → pd1 → out
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


class DecoderPyramid(nn.Module):
    """FPN Top-down 融合 decoder 各层输出（深→浅）"""
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


class UNet_BN_DecoderPyramid(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        # ── Encoder（全部 BNConvBlock）────────────────────────
        self.enc1 = BNConvBlock(input_channels, 64)
        self.enc2 = BNConvBlock(64, 128)
        self.enc3 = BNConvBlock(128, 256)
        self.enc4 = BNConvBlock(256, 512)
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
        c4 = self.enc4(self.pool(c3))
        c5 = self.enc5(self.pool(c4))
        d4 = self.dec4(torch.cat([self.up4(c5), c4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        pd1 = self.dec_pyramid([d4, d3, d2, d1])
        return self.out(pd1)
