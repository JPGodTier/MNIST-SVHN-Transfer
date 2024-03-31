import torch.nn as nn
from src.models.PatchGAN.PatchGANBlocks import ConvBlock


# -----------------------------------------------------------------------------
# PatchGAN
# -----------------------------------------------------------------------------
class PatchGAN(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            # Input: 32x32
            ConvBlock(in_channels, 64, use_instancenorm=False), # 16x16 - no instancenorm to avoid losing information at the first ConvBlock
            ConvBlock(64, 128), # 8x8
            ConvBlock(128, 256), # 4x4
            ConvBlock(256, 512, stride=1), # 3x3 - adjusted stride to prevent too much downsampling
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1), # SVHN: 3x3
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
