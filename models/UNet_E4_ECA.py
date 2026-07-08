"""
UNet_E4_ECA: 基线 UNet + ECA 轻量通道注意力 @ E4（512ch）
1D 卷积捕获跨通道依赖，几乎零额外参数，适合验证通道注意力在 E4 的效果
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ECA(nn.Module):
    """Efficient Channel Attention：1D 卷积捕获跨通道依赖，无降维无参数爆炸"""
    def __init__(self, channels, k=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class UNet_E4_ECA(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        self.enc1 = ConvBlock(input_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.attn = ECA(512)                 # E4 注意力
        self.enc5 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.up4  = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.up3  = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.up2  = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up1  = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.attn(self.enc4(self.pool(c3)))   # ECA @ E4
        c5 = self.enc5(self.pool(c4))
        x = self.dec4(torch.cat([self.up4(c5), c4], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  c3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  c2], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  c1], dim=1))
        return self.out(x)
