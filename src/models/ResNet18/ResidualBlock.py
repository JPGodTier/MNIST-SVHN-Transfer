import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# ResidualBlock
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, skip_conv_bool=True):
        super(ResidualBlock, self).__init__()

        # True if block with convolution in skip connection
        self.skip_conv_bool = skip_conv_bool

        # Define the first 3x3 convolution
        if self.skip_conv_bool:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        # Define the second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Define the 1x1 convolution for the skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        # Define the activation function after the skip connection
        self.relu2 = nn.ReLU()

        
    def forward(self, x):
        # Save the original input for the skip connection (with convolution when required)
        if self.skip_conv_bool:
            shortcut = self.skip_conv(x)
        else:
            shortcut = x

        # First 3x3 conv -> BN -> ReLU
        out = self.relu1(self.bn1(self.conv1(x)))
        
        # Second 3x3 conv -> BN
        out = self.bn2(self.conv2(out))

        
        # Element-wise addition with the shortcut
        out += shortcut
        
        # Final ReLU
        out = self.relu2(out)
        
        return out