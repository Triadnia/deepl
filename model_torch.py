import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)

class PyTorchNet(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(W1.T, dtype=torch.float64))
            self.fc1.bias.copy_(torch.tensor(b1, dtype=torch.float64))
            self.fc2.weight.copy_(torch.tensor(W2.T, dtype=torch.float64))
            self.fc2.bias.copy_(torch.tensor(b2, dtype=torch.float64))
            
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        
    def compute_loss_and_gradients(self, X, y):
        self.zero_grad()
        
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.long)
        
        logits = self(X_t)
        loss = self.criterion(logits, y_t)
        
        loss.backward()
        
        grads = {
            'W1': self.fc1.weight.grad.detach().numpy().T,
            'b1': self.fc1.bias.grad.detach().numpy(),
            'W2': self.fc2.weight.grad.detach().numpy().T,
            'b2': self.fc2.bias.grad.detach().numpy()
        }
        
        return loss.item(), grads