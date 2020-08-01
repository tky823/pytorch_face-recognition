import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class CenterNet(nn.Module):
    def __init__(self, downsample_net, backbone, head_net):
        super().__init__()
        
        self.downsample_net = downsample_net
        self.backbone = backbone
        self.head_net = head_net
        
        self.num_parameters = self._get_num_parameters()
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        """
        x = self.downsample_net(input)
        x = self.backbone(x) # (batch_size, C, H, W)
        output = self.head_net(x)
        
        return output
        
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters


class DownsampleNetBase(nn.Module):
    def __init__(self, channels, kernel_size, stride=None, pool='max'):
        super().__init__()
        
        n_blocks = len(channels) - 1
        
        if type(kernel_size) is not list:
            kernel_size = [kernel_size] * n_blocks
        if stride is None:
            stride = kernel_size
        elif type(stride) is not list:
            stride = [stride] * n_blocks
        
        
        net = []
        
        for n in range(n_blocks):
            net.append(DownsampleBlock(channels[n], channels[n+1], kernel_size=kernel_size[n], stride=stride[n], pool=pool))

        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        output = self.net(input)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, pool='max'):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
        else:
            stride = _pair(stride)
            
        self.kernel_size, self.stride = kernel_size, stride
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        
        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        else:
            raise NotImplementedError("Not support {}".format(pool))
    
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        
        _, _, H, W = input.size()
        padding_height = Kh - 1 - (Sh - (H - Kh) % Sh) % Sh
        padding_width = Kw - 1 - (Sw - (W - Kw) % Sw) % Sw
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        
        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        
        x = self.conv2d(input)
        output = self.pool(x)
        
        return output


if __name__ == '__main__':
    from models.u_net import UNet2d
    from models.head_net import ObjectDetectionNetBase, HeatmapNetBase, LocalOffsetNetBase, SizeNetBase

    batch_size = 4
    
    C = 3
    H, W = 256, 256

    input = torch.randint(0, 5, (batch_size, C, H, W)).float()
    print(input.size())

    in_channels = 4
    channels = [C, in_channels]
    kernel_size, stride = 3, 2
    downsample_net = DownsampleNetBase(channels, kernel_size, stride=stride, pool='max')
    
    channels = [in_channels, 8, 8, 16]
    out_channels = 32
    
    kernel_size, stride, dilated = 3, 1, True
    separable = True
    nonlinear_enc = 'relu'
    nonlinear_dec = ['relu', 'relu', 'relu']
    
    backbone = UNet2d(channels, kernel_size=kernel_size, stride=stride, dilated=dilated, separable=separable, nonlinear_enc=nonlinear_enc, nonlinear_dec=nonlinear_dec, out_channels=out_channels)
    
    kernel_size, stride = 1, 1
    
    head_modules = {
        'heatmap': HeatmapNetBase(out_channels),
        'local_offset': LocalOffsetNetBase(out_channels),
        'size': SizeNetBase(out_channels)
    }
    
    head_net = ObjectDetectionNetBase(head_modules=head_modules)

    center_net = CenterNet(downsample_net, backbone, head_net)
    print(center_net)
    print("# Parameters:", center_net.num_parameters)

    output = center_net(input)
    
    for head_key in head_modules.keys():
        print(head_key, output[head_key].size())
