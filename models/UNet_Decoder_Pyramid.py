"""
UNet_Decoder_Pyramid: 在 decoder 输出侧做 FPN top-down 融合

架构流程
─────────
1. Encoder（标准 UNet，skip connection 使用原始特征 c1-c4）：
   c1(64,224²) → c2(128,112²) → c3(256,56²) → c4(512,28²) → c5(1024,14²)

2. Decoder（标准 UNet）：
   d4(512,28²) ← cat[up(c5), c4]
   d3(256,56²) ← cat[up(d4), c3]
   d2(128,112²)← cat[up(d3), c2]
   d1(64,224²) ← cat[up(d2), c1]

3. DecoderPyramid（新增，decoder 输出侧 FPN）：
   pd4 = smooth(lateral(d4))
   pd3 = smooth(lateral(d3) + top_down(upsample(pd4)))
   pd2 = smooth(lateral(d2) + top_down(upsample(pd3)))
   pd1 = smooth(lateral(d1) + top_down(upsample(pd2)))

4. output = Conv(pd1)  —— pd1 保留 d1 的空间精度 + 获得 d4 的解码语义增强
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DecoderPyramid(nn.Module):
    """
    对 decoder 4 个阶段输出做 FPN top-down 融合
    输入：[d4(512,28²), d3(256,56²), d2(128,112²), d1(64,224²)]  深→浅
    输出：pd1(64,224²)  —— 最浅层融合了所有深层解码语义
    """
    def __init__(self):
        super().__init__()
        chs = [512, 256, 128, 64]
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, c, 1, bias=False) for c in chs
        ])
        self.top_down = nn.ModuleList([
            nn.Conv2d(chs[i], chs[i + 1], 1, bias=False) for i in range(3)
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for c in chs
        ])

    def forward(self, features):
        d4, d3, d2, d1 = features
        pd4 = self.smooth[0](self.lateral[0](d4))
        pd3 = self.smooth[1](
            self.lateral[1](d3) +
            self.top_down[0](F.interpolate(pd4, size=d3.shape[2:], mode='bilinear', align_corners=False))
        )
        pd2 = self.smooth[2](
            self.lateral[2](d2) +
            self.top_down[1](F.interpolate(pd3, size=d2.shape[2:], mode='bilinear', align_corners=False))
        )
        pd1 = self.smooth[3](
            self.lateral[3](d1) +
            self.top_down[2](F.interpolate(pd2, size=d1.shape[2:], mode='bilinear', align_corners=False))
        )
        return pd1


class UNet_Decoder_Pyramid(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        # ── Encoder ─────────────────────────────────────
        self.enc1 = ConvBlock(input_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        # ── Decoder ─────────────────────────────────────
        self.up4  = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.up3  = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.up2  = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up1  = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)
        # ── Decoder-side Pyramid ─────────────────────────
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
