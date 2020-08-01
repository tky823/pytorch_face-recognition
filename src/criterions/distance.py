import torch
import torch.nn as nn

EPS=1e-9

class L1Loss(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        
        self.eps = EPS
        
    def forward(self, input, target, pointmap, n_objects, batch_mean=True):
        """
        Args:
            input: (batch_size, 2, H, W)
            target: (batch_size, 2, H, W)
            point_map: (batch_size, H, W)
            n_objects: (batch_size, )
        """
        coeff = n_objects / (n_objects**2+EPS)
        coeff = coeff.unsqueeze(dim=1) # (batch_size, 1)
        
        loss = pointmap.unsqueeze(dim=1) * torch.abs(input-target) # (batch_size, 2, H, W)
        loss = loss.sum(dim=3).sum(dim=2) # (batch_size, 2)
        loss = coeff * loss # (batch_size, 2)
        loss = loss.sum(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss

class L2Loss(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        
        self.eps = EPS
        
    def forward(self, input, target, pointmap, n_objects, batch_mean=True):
        """
        Args:
            input: (batch_size, 2, H, W)
            target: (batch_size, 2, H, W)
            point_map: (batch_size, H, W)
            n_objects: (batch_size, )
        """
        coeff = n_objects / (n_objects**2+EPS)
        coeff = coeff.unsqueeze(dim=1) # (batch_size, 1)
        
        loss = pointmap.unsqueeze(dim=1) * ((input-target)**2) # (batch_size, 2, H, W)
        loss = loss.sum(dim=3).sum(dim=2) # (batch_size, 2)
        loss = coeff * loss # (batch_size, 2)
        loss = loss.sum(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
