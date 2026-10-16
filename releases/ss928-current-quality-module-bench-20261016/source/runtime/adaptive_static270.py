"""Learn local strength for an existing causal static-detail proposal."""
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn

CACHE270=Path('/tmp/rawir-static270-20261016')

def features270(image,previous,delta,activity,occupancy):
    hp=image-cv2.blur(image,(9,9),borderType=cv2.BORDER_REFLECT)
    broad=image-cv2.blur(image,(21,21),borderType=cv2.BORDER_REFLECT)
    return np.stack([image/255,hp/8,broad/16,delta/3,np.abs(delta)/3,activity,occupancy,(image-previous)/8]).clip(-4,4).astype(np.float16)

class AdaptiveStatic270(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(8,24,3,padding=1,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=2,dilation=2,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=4,dilation=4,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=8,dilation=8,padding_mode='reflect'),nn.ReLU(),nn.Conv2d(24,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-1.0986122886681098)
    def forward(self,features,delta):
        strength=2*torch.sigmoid(self.net(features.clamp(-4,4)))
        return (strength-.5)*delta
