"""Keep the successful instantaneous repair frozen; train only causal gates."""
import torch
from torch import nn
from repair_motion15 import Repair, features

class Repair17(nn.Module):
    def __init__(self,parent=None):
        super().__init__()
        parent=Repair() if parent is None else parent
        self.body=nn.Sequential(*list(parent.net.children())[:-1]).requires_grad_(False)
        self.correct=nn.Conv2d(24,1,3,padding=1).requires_grad_(False)
        self.gate=nn.Conv2d(24,2,3,padding=1)
        with torch.no_grad():
            self.correct.weight.copy_(parent.net[-1].weight[:1]);self.correct.bias.copy_(parent.net[-1].bias[:1])
            self.gate.weight.copy_(parent.net[-1].weight[1:]);self.gate.bias.copy_(parent.net[-1].bias[1:])
    def forward(self,v,previous,previous_output,initial=False):
        with torch.no_grad():
            latent=self.body(features(v,previous))
            corrected=v['baseline']+16*torch.tanh(self.correct(latent))
        heads=self.gate(latent)
        motion=torch.sigmoid(heads[:,1:2]);carry=.95*torch.sigmoid(heads[:,0:1])*(1-motion)
        return corrected if initial else corrected*(1-carry)+previous_output*carry,motion,carry,corrected
