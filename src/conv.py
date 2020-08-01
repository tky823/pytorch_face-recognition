import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

"""
Bottleneck architecture
"""

class BottleneckConv2d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
    
        return output


"""
Depthwise Separable Convolution
"""

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1):
        super().__init__()
        
        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, input):
        x = self.depthwise_conv1d(input)
        output = self.pointwise_conv1d(x)
        
        return output


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
            
        if stride is None:
            stride = kernel_size

        stride = _pair(stride)
        dilation = _pair(dilation)
    
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, input):
        x = self.depthwise_conv2d(input)
        output = self.pointwise_conv2d(x)
        
        return output

"""
Depthwise Separable Transposed Convolution
"""

class DepthwiseSeparableConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.pointwise_conv2d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.depthwise_conv2d = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels)
        

    def forward(self, input):
        x = self.pointwise_conv1d(input)
        output = self.depthwise_conv1d(input)
        
        return output

class DepthwiseSeparableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
            
        if stride is None:
            stride = kernel_size

        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.pointwise_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.depthwise_conv2d = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels)
        

    def forward(self, input):
        x = self.pointwise_conv2d(input)
        output = self.depthwise_conv2d(x)
        
        return output


"""
Partial convolution
"""

class PartialConv2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        return output


"""
Gated convolution
"""

class GatedConv2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        return output
