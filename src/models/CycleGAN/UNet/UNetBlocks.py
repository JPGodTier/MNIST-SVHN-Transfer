import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# DoubleConv
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights for Conv2d layers
        for layer in self.double_conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.double_conv(x)


# -----------------------------------------------------------------------------
# Down
# -----------------------------------------------------------------------------
class Down(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# -----------------------------------------------------------------------------
# Up
# -----------------------------------------------------------------------------
class Up(nn.Module):    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Use bilinear upsampling if specified, otherwise use transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            torch.nn.init.normal_(self.up.weight, mean=0.0, std=0.02)  # Initialize weight for ConvTranspose2d

        
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # Skip connection
        return self.conv(x)


# -----------------------------------------------------------------------------
# OutConv
# -----------------------------------------------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Initialize weights for Conv2d layer
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.conv(x)
