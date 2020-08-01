import torch
import torch.nn as nn

from criterions.distance import L1Loss
from criterions.entropy import BinaryCrossEntropy, FocalLoss

EPS=1e-9

class ObjectDetectionLoss(nn.Module):
    def __init__(self, importance, heatmap_loss):
        """
        Args:
            importance: {'heatmap': , 'local_offset': , 'size': }
        """
        super().__init__()
        
        self.importance = importance
        
        criterions = {}
        
        if heatmap_loss == 'bce':
            criterions['heatmap'] = BinaryCrossEntropy()
        elif heatmap_loss == 'focal':
            criterions['heatmap'] = FocalLoss()
        else:
            raise ValueError("Not support {}".format(heatmap_loss))
            
        criterions['local_offset'] = L1Loss()
        criterions['size'] = L1Loss()

        self.criterions = nn.ModuleDict(criterions)


    def forward(self, input, target, pointmap, n_objects, batch_mean=True, oracle=True):
        """
        Args:
            input: {
                    'heatmap': (batch_size, 1, H, W),
                    'local_offset': (batch_size, 2, H, W),
                    'size': (batch_size, 2, H, W)
                }
            target: {
                'heatmap': (batch_size, H, W),
                'local_offset': (batch_size, 2, H, W),
                'size': (batch_size, 2, H, W)
                }
            n_objects: (batch_size, )
        Returns:
            
            
        """
        importance = self.importance
        
        loss = 0
        
        if oracle:
            for key in importance.keys():
                loss += importance[key] * self.criterions[key](input[key], target[key], pointmap, n_objects, batch_mean=batch_mean)
        else:
            estimated_heatmap = target['heatmap']
            
            raise NotImplementedError("Not implemented estimation criterion.")
        
        return loss
        
