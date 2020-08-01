import torch
import torch.nn as nn

from pool import GlobalMaxPool2d, GlobalAvgPool2d

class SEBlock2d(nn.Module):
    """
        Squeeze-and-extraction block
        See https://arxiv.org/abs/1709.01507
    """
    def __init__(self, in_channels, ratio, pool='average', nonlinear='relu'):
        super().__init__()
        
        if pool=='average':
            self.global_pool = GlobalAvgPool2d(keepdim=True)
        elif pool=='max':
            self.global_pool = GlobalMaxPool2d(keepdim=True)
        else:
            raise NotImplementedError("Not support {}".format(nonlinear))
        
        self.down_fc = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(1,1))
        
        if nonlinear=='relu':
            self.nonlinear = nn.ReLU()
        else:
            raise NotImplementedError("Not support {}".format(nonlinear))
            
        self.up_fc = nn.Conv2d(in_channels//ratio, in_channels, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W): input tensor
        Returns:
            output (batch_size, C, H, W): output tensor whose shape is same as input
        """
        x = self.global_pool(input)
        x = self.down_fc(x)
        x = self.nonlinear(x)
        x = self.up_fc(x)
        output = input * self.sigmoid(x)
        
        return output
        
if __name__ == '__main__':
    torch.manual_seed(111)
    
    batch_size, in_channels, height, width = 4, 128, 32, 64
    ratio = 8

    input = torch.randint(0, 10, (batch_size, in_channels, height, width), dtype=torch.float)
    print(input.size())
    
    se_net = SEBlock2d(in_channels, ratio, pool='average', nonlinear='relu')
    print(senet)
    output = senet(input)
    print(output.size())
