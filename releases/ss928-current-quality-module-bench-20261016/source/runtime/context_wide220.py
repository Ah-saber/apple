"""Increase spatial context while preserving the initialized local function."""
import torch
from torch import nn
from context_refine213 import ContextRefine213

class ContextWide220(ContextRefine213):
    def __init__(self):
        super().__init__()
        old=self.net
        extra=[]
        for dilation in [8,16]:
            conv=nn.Conv2d(32,32,3,padding=dilation,dilation=dilation,padding_mode='reflect')
            with torch.no_grad():
                conv.weight.zero_();conv.bias.zero_()
                for channel in range(32):conv.weight[channel,channel,1,1]=1
            extra.extend([conv,nn.ReLU()])
        self.net=nn.Sequential(*list(old.children())[:6],*extra,old[6])

    def initialize_local(self,state):
        mapped={}
        for key,value in state.items():
            mapped[key.replace('net.6.','net.10.')]=value
        self.load_state_dict(mapped,strict=False)
