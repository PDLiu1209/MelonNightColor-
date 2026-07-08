import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Unet, self).__init__()

        def conv_block(input_channels, num_classes):
            return nn.Sequential(
                nn.Conv2d(input_channels, num_classes, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_conv(input_channels, num_classes):
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, num_classes, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        # Down-sampling path
        self.conv1 = conv_block(input_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Up-sampling path
        self.up4 = up_conv(1024, 512)
        self.conv_up4 = conv_block(1024, 512)

        self.up3 = up_conv(512, 256)
        self.conv_up3 = conv_block(512, 256)

        self.up2 = up_conv(256, 128)
        self.conv_up2 = conv_block(256, 128)

        self.up1 = up_conv(128, 64)
        self.conv_up1 = conv_block(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Down-sampling
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # Up-sampling
        u4 = self.up4(c5)
        cat4 = torch.cat([u4, c4], dim=1)
        cu4 = self.conv_up4(cat4)

        u3 = self.up3(cu4)
        cat3 = torch.cat([u3, c3], dim=1)
        cu3 = self.conv_up3(cat3)

        u2 = self.up2(cu3)
        cat2 = torch.cat([u2, c2], dim=1)
        cu2 = self.conv_up2(cat2)

        u1 = self.up1(cu2)
        cat1 = torch.cat([u1, c1], dim=1)
        cu1 = self.conv_up1(cat1)

        # Final output with Sigmoid activation
        output = self.final_conv(cu1)

        return output


# Example usage:
if __name__ == "__main__":
    model = Unet(input_channels=3, num_classes=1)  # For binary segmentation
    x = torch.randn(4, 3, 224, 224)  # Input tensor
    output = model(x)
    print(output.shape)  # Output shape
