"""Allow larger shape changes within causal current RAW foreground ownership."""
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from tail_context284 import TailContext284

CACHE300=Path('/tmp/rawir-shape300-20261016')
def owner300(compact,active):
    return cv2.GaussianBlur(np.maximum(compact,active).astype(np.float32),(3,3),.6).clip(0,1).astype(np.float16)
def features300(feature,owner):return np.concatenate([feature,owner[None]],axis=0).astype(np.float16)

class ForegroundShape300(TailContext284):
    def __init__(self):
        super().__init__();self.net[0]=nn.Conv2d(24,32,3,padding=1,padding_mode='reflect')
    def initialize_shape(self,state):
        with torch.no_grad():
            self.net[0].weight[:,:23].copy_(state['net.0.weight']);self.net[0].weight[:,23:].zero_()
        self.load_state_dict({k:v for k,v in state.items() if k!='net.0.weight'},strict=False)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,feature,mask):
        amount=8+24*feature[:,23:24].clamp(0,1)
        return amount*torch.tanh(self.net(feature.clamp(-4,4)))*mask
