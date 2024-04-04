import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        # Input: 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=2)  # Output: 64x17x17
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2)  # Output: 128x9x9
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2)  # Output: 256x5x5
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2)  # Output: 512x3x3
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=2)  # Output: 512x2x2
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=2, stride=1, padding=0)  # Output: 1x1x1

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(F.instance_norm(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(F.instance_norm(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(F.instance_norm(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(F.instance_norm(self.conv5(x)), negative_slope=0.2)
        probabilities = F.sigmoid(self.conv6(x))
        return probabilities