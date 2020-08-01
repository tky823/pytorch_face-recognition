import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

"""
    Global max pooling
"""

class GlobalMaxPool2d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        
        self.keepdim = keepdim
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, 1, 1)
        """
        _, _, H, W = input.size()
        
        output = F.max_pool2d(input, kernel_size=(H, W))
        
        if not self.keepdim:
            output = output.squeeze(dim=3).squeeze(dim=2)
        
        return output


"""
    Global average pooling
"""

class GlobalAvgPool2d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        
        self.keepdim = keepdim
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, 1, 1)
        """
        _, _, H, W = input.size()
        
        output = F.avg_pool2d(input, kernel_size=(H, W))
        
        if not self.keepdim:
            output = output.squeeze(dim=3).squeeze(dim=2)
        
        return output

"""
    Lp norm pooling
"""

class LpNormPool2d(nn.Module):
    def __init__(self, num_features, kernel_size, stride=None, p=None):
        super().__init__()
        
        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size
            
        stride = _pair(stride)
        
        self.num_features = num_features
        self.kernel_size, self.stride = kernel_size, stride
        
        if p is None:
            self.p = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.p = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
            self.p.data.fill_(p)
        self.c = nn.Parameter(torch.zeros((num_features,)+kernel_size), requires_grad=True)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, H', W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        """
        batch_size, C, H, W = input.size()
        
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        
        p = self.p.view(C, 1, 1)
        c = self.c.view(C, Kh*Kw, 1)
        input = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C*Kh*Kw, H'*W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        input = input.view(batch_size, C, Kh*Kw, -1) # -> (batch_size, C, Kh*Kw, H'*W')
        
        x = torch.abs(input - c)**p # -> (batch_size, C, Kh*Kw, H'*W')
        x = x.mean(dim=2, keepdim=True) # -> (batch_size, C, 1, H'*W')
        output = x**(1/p) # -> (batch_size, C, 1, H'*W')
        output = output.squeeze(dim=2)
        output = F.fold(output, kernel_size=(1,1), stride=self.stride, output_size=((H-Kh)//Sh+1,(W-Kw)//Sw+1))
    
        return output
        
"""
    Mixed pooling
"""
class MixedPool2d(nn.Module):
    """
        See "Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree"
        https://arxiv.org/abs/1509.08985
    """
    def __init__(self, kernel_size, stride=None, max_pool_weight=None):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
        
        stride = _pair(stride)
            
        self.kernel_size, self.stride = kernel_size, stride
         
        if max_pool_weight is None:
            self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        else:
            self.a = nn.Parameter(torch.tensor(max_pool_weight), requires_grad=False)
            
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, H', W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        """
        kernel_size, stride = self.kernel_size, self.stride
        
        x_max = F.max_pool2d(input, kernel_size=kernel_size, stride=stride)
        x_avg = F.avg_pool2d(input, kernel_size=kernel_size, stride=stride)
        output = self.a * x_max + (1 - self.a) * x_avg
        
        return output

"""
    Gated pooling
"""

class GatedPool2d(nn.Module):
    """
        See "Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree"
        https://arxiv.org/abs/1509.08985
    """
    def __init__(self, num_features, kernel_size, stride=None):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
        
        if stride is None:
            stride = kernel_size
            
        stride = _pair(stride)
            
        self.kernel_size, self.stride = kernel_size, stride
         
        size = (num_features,) + kernel_size
        self.weight = nn.Parameter(torch.Tensor(*size), requires_grad=True)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, H', W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        
        batch_size, _, H, W = input.size()
        
        x_max = F.max_pool2d(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C, H',W')
        x_avg = F.avg_pool2d(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C, H',W')
        
        input = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C*Kh*Kw, H'*W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        weight = self.weight.view(-1, 1) # -> (C*Kh*Kw, 1)
        x = weight * input # -> (batch_size, C*Kh*Kw, H'*W')
        x = x.sum(dim=1).view(batch_size, 1, (H-Kh)//Sh+1, (W-Kw)//Sw+1) # -> (batch_size, 1, H',W')
        mask = torch.sigmoid(x)
        
        output = mask * x_max + (1 - mask) * x_avg
        
        return output
            
"""
    Stochatic pooling
"""

class StochaticPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        
        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size
            
        stride = _pair(stride)
            
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, H', W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        """
        batch_size, C, H, W = input.size()
        
        if (input < 0).any():
            raise ValueError("Place non-negative output function before Stochatic pooling.")
        elif (input==0).all():
            zero_flg = 1
        else:
            zero_flg = 0
        
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        
        input = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C*Kh*Kw, H'*W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        input = input.view(batch_size*C, Kh*Kw, -1) # -> (batch_size*C, Kh*Kw, H'*W')
        input = input.permute(0,2,1).contiguous() # -> (batch_size*C, H'*W', Kh*Kw)
        input = input.view(-1, Kh*Kw) # -> (batch_size*C*H'*W', Kh*Kw)
        
        weights = input # (batch_size*C*H'*W', Kh*Kw)
        
        if zero_flg:
            weights = weights + 1
            # So weights' elements are all 1.

        if self.training:
            indices = torch.multinomial(weights, 1).view(-1) # -> (batch_size*C*H'*W', 1)
            mask = torch.eye(Kh*Kw)[indices].float() # -> (batch_size*C*H'*W', Kh*Kw)
            output = mask * input
        else:
            weights /= weights.sum(dim=1, keepdim=True) # -> (batch_size*C*H'*W', Kh*Kw)
            output = weights * input # -> (batch_size*C*H'*W', Kh*Kw)

        output = output.sum(dim=1) # -> (batch_size*C*H'*W',)
        output = output.view(batch_size, C, -1) # -> (batch_size, C, H'*W')
        output = F.fold(output, kernel_size=(1,1), stride=self.stride, output_size=((H-Kh)//Sh+1,(W-Kw)//Sw+1))
        
        return output

if __name__ == '__main__':
    torch.manual_seed(111)
    
    batch_size, num_features, height, width = 1, 2, 3, 4
    kernel_size = (2,3)
    stride = (1,1)

    input = torch.randint(0, 10, (batch_size, num_features, height, width), dtype=torch.float)
    print(input)
    print(input.size())
    
    # Global max pooling
    print("Global max pooling")
    global_max_pool2d = GlobalMaxPool2d()
    output = global_max_pool2d(input)
    print(output)
    print(output.size())
    print()
    
    # Global average pooling
    print("Global average pooling")
    global_avg_pool2d = GlobalAvgPool2d()
    output = global_avg_pool2d(input)
    print(output)
    print(output.size())
    print()
    
    # Lp norm pooling
    print("Lp norm pooling")
    lp_norm_pool2d = LpNormPool2d(num_features=num_features, kernel_size=kernel_size, stride=(1,1))
    output = lp_norm_pool2d(input)
    print(output)
    print(output.size())
    print()
    
    # Mixed pooling
    print("Mixed pooling")
    mixed_pool2d = MixedPool2d(kernel_size=kernel_size, stride=(1,1))
    output = mixed_pool2d(input)
    print(output)
    print(output.size())
    print()
    
    # Gated pooling
    print("Gated pooling")
    gated_pool2d = GatedPool2d(num_features=num_features, kernel_size=kernel_size, stride=(1,1))
    output = gated_pool2d(input)
    print(output)
    print(output.size())
    print()
    
    # Stochatic pooling
    print("Stochatic pooling")
    stochatic_pool2d = StochaticPool2d(kernel_size=kernel_size, stride=(1,1))
    output = stochatic_pool2d(input)
    print(output)
    print(output.size())
    
    stochatic_pool2d.eval()
    output = stochatic_pool2d(input)
    print(output)
    print(output.size())
    print()
