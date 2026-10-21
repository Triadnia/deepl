import torch
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(29, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out