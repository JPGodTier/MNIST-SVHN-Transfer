import torch.nn as nn
import torch.nn.functional as F

    
# -----------------------------------------------------------------------------
# ClassPredictor
# -----------------------------------------------------------------------------
class ClassPredictor(nn.Module):
    def __init__(self, prob=0.5):
        super(ClassPredictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        #if reverse:
            #x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x