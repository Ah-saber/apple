"""Expand compact-target context while retaining the trained signed correction."""
import torch
from torch import nn
from small_context227 import SmallContext227

class SmallWide244(SmallContext227):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(*list(self.net.children())[:-1],nn.Conv2d(32,32,3,padding=32,dilation=32,padding_mode='reflect'),nn.ReLU(),list(self.net.children())[-1])
    def initialize_wide(self,state):
        mapped={('net.12.'+key.split('.')[-1] if key.startswith('net.10.') else key):value for key,value in state.items()}
        self.load_state_dict(mapped,strict=False)
        with torch.no_grad():
            self.net[10].weight.zero_();self.net[10].bias.zero_()
            for channel in range(32):self.net[10].weight[channel,channel,1,1]=1
