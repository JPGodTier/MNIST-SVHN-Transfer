import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskClassifier(nn.Module):
    def __init__(self):
        super(TaskClassifier, self).__init__()
        # Input: 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=0)  # Output: 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 14x14
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)  # Output: 10x10
        self.dropoutconv2 = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 5x5
        self.fc1 = nn.Linear(in_features=50 * 5 * 5, out_features=500)  # Output: 500 (flattened)
        self.dropoutfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=500, out_features=10)  # Output: 10 (scores)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.dropoutconv2(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor from start_dim=1
        x = self.dropoutfc1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x