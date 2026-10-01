"""Freeze reference coordinates for the review-only reliable-history prototype."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from compact_model import FixedSample,Rows32


class BoardReliable(nn.Module):
    def __init__(self,source,box,layout='rows32'):
        super().__init__();self.model=copy.deepcopy(source);self.layout=layout
        self.sample=FixedSample(box,256,320)
        self.resizes=nn.ModuleList([FixedSample(torch.tensor([[0.,0.,1.,1.]]),64,64,64//s) for s in (2,4,8)])
        self.output=Rows32()

    def forward(self,x,context):
        m=self.model
        current,mean,history,reliability=m.components(x)
        features=F.relu(m.front(torch.cat((current,mean,history),1)))
        residual=m.head(m.body(features))
        ref=m.reference.encoder(context.to(features.dtype))
        if hasattr(m.reference,'pyramid'):
            for i,branch in enumerate(m.reference.pyramid):
                ref=ref+self.resizes[i](branch(context.to(features.dtype)).float()).to(ref.dtype)
        ref=m.reference.project(ref+ref.mean((-2,-1),keepdim=True))
        reference=self.sample(ref.float()).to(features.dtype)
        phases=(current+residual+reference).clamp(0,1)*255
        return phases if self.layout=='phases' else self.output(phases)
