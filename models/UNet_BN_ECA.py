"""
UNet_BN_ECA: BN（全局归一化）+ ECA @ E4（轻量通道注意力）

组合逻辑
────────
  UNet_BN   : mIoU 0.8677  StemIoU 0.6745  (+0.0099 / +0.0214) ← 新最优
  UNet_E4_ECA: mIoU 0.8623  StemIoU 0.6662  (+0.0045 / +0.0131)

两者改进机制完全正交：
  BN   → 改善激活分布、正则化、使优化器收敛到更好极值
  ECA  → 通道注意力，让网络显式聚焦茎部相关通道

在 BN 规范化的激活空间上，ECA 的通道注意力权重应更稳定、更准确，
有望实现 +0.0099 + α 的叠加增益。

架构流程
─────────
  enc1→enc2→enc3→enc4(BN)→ECA→enc5(BN)→dec4(BN)→dec3(BN)→dec2(BN)→dec1(BN)→out
"""
import torch
import torch.nn as nn


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


class UNet_BN_ECA(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        # ── Encoder（全部 BNConvBlock）────────────────────────
        self.enc1 = BNConvBlock(input_channels, 64)
        self.enc2 = BNConvBlock(64, 128)
        self.enc3 = BNConvBlock(128, 256)
        self.enc4 = BNConvBlock(256, 512)
        self.eca  = ECA(512)                    # ECA @ E4
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
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.eca(self.enc4(self.pool(c3)))    # BN + ECA @ E4
        c5 = self.enc5(self.pool(c4))
        x = self.dec4(torch.cat([self.up4(c5), c4], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  c3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  c2], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  c1], dim=1))
        return self.out(x)
