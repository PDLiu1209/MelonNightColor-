"""
UNet_BN: 基线 UNet + BatchNorm（批归一化）in all ConvBlocks

动机
────
基础 UNet 的 ConvBlock：
  Conv → ReLU → Conv → ReLU （无任何归一化）

BatchNorm 是现代深度学习最基础的组件之一，本项目 UNet 竟完全缺失：

  Conv → BN → ReLU → Conv → BN → ReLU

BN 的三大作用：
  ① 归一化激活分布 → 缓解内部协变量偏移（Internal Covariate Shift）
  ② 正则化效果 → 等效于对 batch 内样本的随机扰动，防止过拟合
  ③ 使更高学习率可用 → 更快收敛

注意事项：
  - Conv bias=False（BN 的 beta 参数已包含偏置功能，bias 冗余）
  - UpConv 中不加 BN（转置卷积后直接 skip concat，BN 可能引入 artifact）

期望：BN 缺失是最基础的工程问题，补上后预计综合 mIoU 有显著提升
额外参数：约 +0.01M（BN 每层仅 2C 个参数，可忽略）
"""
import torch
import torch.nn as nn


class BNConvBlock(nn.Module):
    """带 BatchNorm 的双层卷积块"""
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


class UNet_BN(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        # ── Encoder ─────────────────────────────────────
        self.enc1 = BNConvBlock(input_channels, 64)
        self.enc2 = BNConvBlock(64, 128)
        self.enc3 = BNConvBlock(128, 256)
        self.enc4 = BNConvBlock(256, 512)
        self.enc5 = BNConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        # ── Decoder ─────────────────────────────────────
        self.up4  = UpConv(1024, 512)
        self.dec4 = BNConvBlock(1024, 512)
        self.up3  = UpConv(512, 256)
        self.dec3 = BNConvBlock(512, 256)
        self.up2  = UpConv(256, 128)
        self.dec2 = BNConvBlock(256, 128)
        self.up1  = UpConv(128, 64)
        self.dec1 = BNConvBlock(128, 64)
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        c5 = self.enc5(self.pool(c4))
        x = self.dec4(torch.cat([self.up4(c5), c4], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  c3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  c2], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  c1], dim=1))
        return self.out(x)
