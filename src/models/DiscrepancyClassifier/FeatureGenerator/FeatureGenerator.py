import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# FeatureGenerator
# -----------------------------------------------------------------------------
class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


# -----------------------------------------------------------------------------
# FeatureGeneratorSmall
# -----------------------------------------------------------------------------
class FeatureGeneratorSmall(nn.Module):
    def __init__(self):
        super(FeatureGeneratorSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2) # Output: 32x32
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2) # Output: 16x16
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2) # Output: 8x8
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(8*8*16, 512)
        self.bn1_fc = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 1024)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x
