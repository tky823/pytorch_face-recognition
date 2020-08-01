import torch
import torch.nn as nn

def shuffle(input):
    """
    Args:
        input (batch_size, C, *): input tensor
    Returns:
        output (batch_size, C, *): output tensor whose shape is same as input
    """
    C = input.size()[1]
    shuffled_channel = torch.randperm(C)
    
    output = input[:,shuffled_channel]
    
    return output

class ShuffleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, *): input tensor
        Returns:
            output (batch_size, C, *): output tensor whose shape is same as input
        """
        output = shuffle(input)
        
        return output
        
if __name__ == '__main__':
    torch.manual_seed(111)
    
    batch_size, in_channels, height, width = 2, 4, 3, 2

    input = torch.randint(0, 10, (batch_size, in_channels, height, width), dtype=torch.float)
    print(input)
    print(input.size())
    
    shuffle_net = ShuffleBlock()
    
    output = shuffle_net(input)
    print(output)
    print(output.size())

