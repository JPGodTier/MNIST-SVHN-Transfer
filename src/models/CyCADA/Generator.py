import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input: 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3) # Output: 16x16
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # Output: 8x8
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1) # Output: 4x4
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # Output: 4x4
        self.conv42 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # Output: 4x4
        self.conv51 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # Output: 4x4
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # Output: 4x4
        self.conv6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 8x8
        self.conv7 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 16x16
        self.conv8 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3) # Output: 32x32

    def forward(self, x):
        x = F.relu(F.instance_norm(self.conv1(x)))
        x = F.relu(F.instance_norm(self.conv2(x)))
        x = F.relu(F.instance_norm(self.conv3(x)))

        y = F.relu(F.instance_norm(self.conv41(x)))
        y = F.instance_norm(self.conv42(y))
        y = y + x # Adding residual

        z = F.relu(F.instance_norm(self.conv51(y)))
        z = F.instance_norm(self.conv52(z))
        z = z + y # Adding residual

        z = F.relu(F.instance_norm(self.conv6(z)))
        z = F.relu(F.instance_norm(self.conv7(z)))
        output = F.tanh(self.conv8(z))
        return output