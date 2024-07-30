import torch
import torch.nn as nn

class MeanEuclideanDistance(nn.Module):
    def __init__(self):
        super(MeanEuclideanDistance, self).__init__()
    
    def forward(self, y_pred, y_true):
        return torch.mean(torch.linalg.norm(torch.sub(y_true, y_pred), dim=1))