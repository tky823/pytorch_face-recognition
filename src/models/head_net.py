import torch
import torch.nn as nn
import torch.nn.functional as F

_head_modules = {
    'heatmap': None,
    'local_offset': None,
    'size': None
}

class ObjectDetectionNetBase(nn.Module):
    def __init__(self, head_modules=_head_modules):
        super().__init__()
        
        head_keys = set(head_modules.keys())

        if set(_head_modules.keys()) - head_keys != set():
            raise ValueError("head_modules are insufficient.")
        
        self.head_keys = head_keys
        self.net = nn.ModuleDict(head_modules)
        
    def forward(self, input):
        output = {}
        
        for head_key in self.head_keys:
            output[head_key] = self.net[head_key](input)
        
        return output
        
class HeatmapNetBase(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1)
        self.nonlinear = nn.Sigmoid()
        
    def forward(self, input):
        x = self.conv2d(input)
        output = self.nonlinear(x)
        
        return output
        
class LocalOffsetNetBase(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels=2, kernel_size=1, stride=1)
        
    def forward(self, input):
        output = self.conv2d(input)
        
        return output
        
class SizeNetBase(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels=2, kernel_size=1, stride=1)
        self.nonlinear = nn.ReLU()
        
    def forward(self, input):
        x = self.conv2d(input)
        output = self.nonlinear(x)
        
        return output
        

if __name__ == '__main__':
    in_channels = 32
    kernel_size, stride = 1, 1
    head_modules = {
        'heatmap': HeatmapNetBase(in_channels),
        'local_offset': LocalOffsetNetBase(in_channels),
        'size': SizeNetBase(in_channels)
    }
    
    head_net = ObjectDetectionNetBase(head_modules=head_modules)
    
    print(head_net)
