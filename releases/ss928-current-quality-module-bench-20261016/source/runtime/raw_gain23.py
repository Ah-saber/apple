"""A signed current/background difference can restore targets and departures."""
import torch
from torch import nn
from raw_route22 import RawRoute22
from repair_motion15 import features

class RawGain23(nn.Module):
    def __init__(self,parent=None):
        super().__init__();parent=RawRoute22() if parent is None else parent
        self.body=parent.body.requires_grad_(False);self.correct=parent.correct.requires_grad_(False)
        self.gate=nn.Conv2d(24,4,3,padding=1)
        with torch.no_grad():
            self.gate.weight[:3].copy_(parent.gate.weight);self.gate.bias[:3].copy_(parent.gate.bias)
            self.gate.weight[3:].zero_();self.gate.bias[3:].zero_()
    def forward(self,v,previous,previous_output,initial=False):
        with torch.no_grad():
            latent=self.body(features(v,previous));fixed=v['baseline']+16*torch.tanh(self.correct(latent))
        heads=self.gate(latent)
        corrected=fixed+torch.tanh(heads[:,2:3])*(v['current_only']-v['baseline'])
        corrected=corrected+torch.tanh(heads[:,3:4])*(v['current_only']-v['past_background'])
        motion=torch.sigmoid(heads[:,1:2]);carry=.95*torch.sigmoid(heads[:,0:1])*(1-motion)
        return corrected if initial else corrected*(1-carry)+previous_output*carry,motion,carry,corrected
