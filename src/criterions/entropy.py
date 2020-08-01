import torch
import torch.nn as nn

EPS=1e-9

class BinaryCrossEntropy(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        
        self.eps = eps
        
    def forward(self, input, target, pointmap, n_objects, batch_mean=True):
        """
        Args:
            input: (batch_size, 1, H, W)
            target: (batch_size, H, W)
        """
        eps = self.eps
        input = input.squeeze(dim=1) # (batch_size, H, W)

        loss = - target*torch.log(input + eps) - (1-target)*torch.log(1 - input + eps)
        loss = loss.sum(dim=1).sum(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=EPS):
        super().__init__()

        self.gamma = gamma
        self.eps = eps
        
    def forward(self, input, target, pointmap, n_objects, batch_mean=True):
        """
        Args:
            input: (batch_size, 1, H, W)
            target: (batch_size, H, W)
        """
        eps = self.eps
        input = input.squeeze(dim=1) # (batch_size, H, W)

        loss = - target*((1 - input)**gamma)*torch.log(input + eps) - (1 - target)*(input**gamma)*torch.log(1 - input + eps)
        loss = loss.sum(dim=1).sum(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
