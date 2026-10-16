"""Signed current-small-body correction, with an explicit physical reference cue."""
from pathlib import Path
import numpy as np,torch
from torch import nn
from context_wide220 import ContextWide220

CACHE227=Path('/tmp/rawir-small227-20261016')

class SmallContext227(ContextWide220):
    def __init__(self):
        super().__init__()
        self.net[0]=nn.Conv2d(19,32,3,padding=1,padding_mode='reflect')
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def initialize_context(self,state):
        with torch.no_grad():
            self.net[0].weight[:,:18].copy_(state['net.0.weight']);self.net[0].weight[:,18:].zero_()
        mapped={key:value for key,value in state.items() if key not in ['net.0.weight','net.10.weight','net.10.bias']}
        self.load_state_dict(mapped,strict=False)
    def forward(self,feature,mask):return 24*torch.tanh(self.net(feature.clamp(-4,4)))*mask

def features227(old,extra,released):
    value=np.asarray(old,np.float32).copy()
    for channel,position in [(0,0),(9,1),(13,2),(16,3)]:value[channel]=extra[position]
    value[17]=np.maximum(value[17],released)
    return np.concatenate([value,np.asarray(extra[4:5],np.float32)],axis=0).clip(-4,4).astype(np.float16)
