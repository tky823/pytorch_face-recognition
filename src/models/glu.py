import torch
import torch.nn as nn
import torch.nn.functional as F
        
class GLU(nn.Module):
    """
    Gated Linear Units
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Linear(in_channels, out_channels)
        self.map_gate = nn.Linear(in_channels, out_channels)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, *)
        Returns:
            output (batch_size, out_channels, *)
        """
        dim = input.dim()
        size = input.size()
        batch_size = size[0]
        reshaped_size = (batch_size, *size[2:], self.in_channels)
        output_size = (batch_size, self.out_channels, *size[2:])
        
        input = input.view(*reshaped_size)
        x = self.map(input)
        x_sigmoid = self.map_gate(input)
        output = x * torch.sigmoid(x_sigmoid)
        output = output.view(*output_size)
        
        return output
