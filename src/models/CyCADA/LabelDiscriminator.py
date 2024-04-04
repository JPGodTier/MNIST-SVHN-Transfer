import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelDiscriminator(nn.Module):
    def __init__(self):
        super(LabelDiscriminator, self).__init__()
        self.fc1 = nn.Linear(10, 500)  # The input features should match the number of classes from TaskClassifier.
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2)  # Outputs 2 scores: one for real and zero for fake.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probabilities = F.softmax(self.fc3(x), dim=1)  # Apply softmax to convert to probabilities
        return probabilities