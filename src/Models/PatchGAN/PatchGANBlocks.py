import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ConvBlock
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_instancenorm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        
        if use_instancenorm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)