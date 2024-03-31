import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ResNet18.ResidualBlock import ResidualBlock

# -----------------------------------------------------------------------------
# ResNet18
# -----------------------------------------------------------------------------
class ResNet18(nn.Module):
    def __init__(self, in_channels=3, stride=2):
        super(ResNet18, self).__init__()

        # Define the first convolution block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()

        # Define the second convolution block
        self.conv2_1 = ResidualBlock(in_channels=64, out_channels=64, stride=stride, skip_conv_bool=False)
        self.conv2_2 = ResidualBlock(in_channels=64, out_channels=64, stride=stride, skip_conv_bool=False)

        # Define the third convolution block
        self.conv3_1 = ResidualBlock(in_channels=64, out_channels=128, stride=stride, skip_conv_bool=True)
        self.conv3_2 = ResidualBlock(in_channels=128, out_channels=128, stride=stride, skip_conv_bool=False)

        # Define the fourth convolution block
        self.conv4_1 = ResidualBlock(in_channels=128, out_channels=256, stride=stride, skip_conv_bool=True)
        self.conv4_2 = ResidualBlock(in_channels=256, out_channels=256, stride=stride, skip_conv_bool=False)

        # Define the fifth convolution block
        self.conv5_1 = ResidualBlock(in_channels=256, out_channels=512, stride=stride, skip_conv_bool=True)
        self.conv5_2 = ResidualBlock(in_channels=512, out_channels=512, stride=stride, skip_conv_bool=False)

        # Define the fully connected block
        self.avgpool6 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc6_1 = nn.Linear(in_features=1*1*512, out_features=1000)
        self.fc6_2 = nn.Linear(in_features=1000, out_features=10)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x):
        # Check if the input x has 1 channel (e.g., MNIST), and if so, convert it to 3 channels
        if x.size(1) == 1: 
            x = x.repeat(1, 3, 1, 1)  # Repeat the channel 3 times

        # Resize the image to 224x224 for ResNet18
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # First convolution block
        out = self.relu1(self.maxpool1(self.bn1(self.conv1(x))))

        # Second convolution block
        out = self.conv2_2(self.conv2_1(out))

        # Third convolution block
        out = self.conv3_2(self.conv3_1(out))

        # Fourth convolution block
        out = self.conv4_2(self.conv4_1(out))

        # Fifth convolution block
        out = self.conv5_2(self.conv5_1(out))

        # Fully connected block
        out = torch.flatten(self.avgpool6(out), 1)
        out = self.softmax6(self.fc6_2(self.fc6_1(out)))

        return out