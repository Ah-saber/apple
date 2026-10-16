"""Preserve the four-route parent while adding the nine physical RAW features."""
import torch
from torch import nn
from raw_gain23 import RawGain23
from raw_temporal24 import RawFeatures24,RawTemporal24

class RawTemporal25(RawTemporal24):
    def __init__(self,parent=None):
        nn.Module.__init__(self)
        parent=RawGain23() if parent is None else parent
        self.body=parent.body;self.correct=parent.correct;self.gate=parent.gate
        old=self.body[0];expanded=nn.Conv2d(19,24,3,padding=1)
        with torch.no_grad():
            expanded.weight[:,:10].copy_(old.weight);expanded.weight[:,10:].zero_();expanded.bias.copy_(old.bias)
        expanded.bias.requires_grad_(False)
        expanded.weight.register_hook(lambda g:torch.cat([torch.zeros_like(g[:,:10]),g[:,10:]],1))
        self.body[0]=expanded
