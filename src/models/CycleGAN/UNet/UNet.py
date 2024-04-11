import torch.nn as nn
from src.models.UNet.UNetBlocks import DoubleConv, Down, Up, OutConv

# -----------------------------------------------------------------------------
# Unet
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(self.in_channels, 64)  # Output size: 32x32
        self.down1 = Down(64, 128)  # Output size: 16x16
        self.down2 = Down(128, 256)  # Output size: 8x8
        self.up1 = Up(256 + 128, 128, bilinear)  # Output size: 16x16
        self.up2 = Up(128 + 64, 64, bilinear)  # Output size: 32x32
        self.outc = OutConv(64, self.out_channels)  # Output size: 32x32

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits
