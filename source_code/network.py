import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, **flags):
        super(Network, self).__init__()
        self.seed = flags['seed']
        self.linear_blocks = self._build_linear_blocks(flags['input_dim'], flags['n_classes'], flags['width'], flags['depth'])
        
    def _build_linear_blocks(self, input_dim, output_dim, width, depth):
        layers = []
        
        # input layer
        layers.append(nn.Linear(input_dim, width, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width, bias=True))
            layers.append(nn.ReLU(inplace=True))
        
        # output layer
        layers.append(nn.Linear(width, output_dim, bias=True))
        
        # initialize as Xavier Normal
        for layer in layers:
            if isinstance(layer, nn.Linear):
                torch.manual_seed(self.seed)
                nn.init.xavier_normal_(layer.weight)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_blocks(x)
    
def free_device(tensors):
    l = []
    for tensor in tensors:
        tensor = tensor.detach().cpu().numpy()
        l.append(tensor)
    return l