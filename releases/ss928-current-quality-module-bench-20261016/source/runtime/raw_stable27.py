"""Stabilize actual background changes; foreground contrast uses current space."""
import torch
from torch.nn import functional as F
from raw_temporal25 import RawTemporal25
from repair_motion15 import features

class RawStable27(RawTemporal25):
    def __init__(self,parent=None):
        if parent is None:
            super().__init__()
        else:
            torch.nn.Module.__init__(self)
            self.body=parent.body;self.correct=parent.correct;self.gate=parent.gate
        self.body[2].requires_grad_(True)
        self.correct.requires_grad_(True)
    def forward(self,v,previous,previous_output,initial=False):
        latent=self.body(torch.cat([features(v,previous),v['raw24']],1))
        heads=self.gate(latent)
        current=v['current_only']
        spatial=current-F.avg_pool2d(current,17,1,8,count_include_pad=False)
        corrected=v['baseline']+16*torch.tanh(self.correct(latent))
        corrected=corrected+torch.tanh(heads[:,2:3])*(current-v['baseline'])+torch.tanh(heads[:,3:4])*spatial
        motion=torch.sigmoid(heads[:,1:2]);carry=.95*torch.sigmoid(heads[:,0:1])*(1-motion)
        return corrected if initial else corrected*(1-carry)+previous_output*carry,motion,carry,corrected
