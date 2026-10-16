"""Bounded local repair with causal observation changes and annulus supervision."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

CACHE=Path('/tmp/rawir-context213-20261015')
DELTA=Path('/tmp/rawir-residual213-20261015')

class ContextRefine213(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(18,32,3,padding=1,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=2,dilation=2,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4,padding_mode='reflect'),nn.ReLU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,mask):
        return 6*torch.tanh(self.net(features.clamp(-4,4)))*mask

def assemble213(old,base,hp,change,raw_change,stable_hint,repair_difference,current_support):
    feature=np.concatenate([np.asarray(old,np.float32),np.stack([change/16,raw_change/16,stable_hint/3,repair_difference/16,current_support])],axis=0)
    feature[0]=base/255;feature[9]=hp/16
    return feature.clip(-4,4).astype(np.float16)
