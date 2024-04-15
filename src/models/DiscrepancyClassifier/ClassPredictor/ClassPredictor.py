import torch.nn as nn
import torch.nn.functional as F
from src.Common.utils import grad_reverse

    
# -----------------------------------------------------------------------------
# ClassPredictor
# -----------------------------------------------------------------------------
class ClassPredictor(nn.Module):
    def __init__(self, lambd=1.0):
        super(ClassPredictor, self).__init__()
        self.fc1 = nn.Linear(3072, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.lambd = lambd

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x


# -----------------------------------------------------------------------------
# ClassPredictorSmall
# -----------------------------------------------------------------------------
class ClassPredictorSmall(nn.Module):
    def __init__(self, lambd=1.0):
        super(ClassPredictorSmall, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        self.lambd = lambd

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x