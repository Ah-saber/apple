"""Zero-initialized, gated target residual with the original denoiser frozen."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from adaptive_history import HistorySignal,fold_current


class MotionResidual(nn.Module):
    def __init__(self,source,kind,threshold):
        super().__init__();self.source=copy.deepcopy(source);self.kind=kind;self.factor=4 if kind=='quarter' else 2
        q=self.q;first=q.front if kind=='quarter' else q.front.first
        self.current=fold_current(first);self.gate=HistorySignal(threshold,self.factor)
        kernel=torch.zeros(self.factor**2,1,self.factor,self.factor)
        for i in range(self.factor**2):kernel[i,0,i//self.factor,i%self.factor]=1
        self.register_buffer('phase_kernel',kernel)
        self.residual=nn.Conv2d(first.out_channels+2*self.factor**2,self.factor**2,1)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        self.source.requires_grad_(False);self.current.requires_grad_(False)

    @property
    def q(self):return self.source if self.kind=='quarter' else self.source.quarter

    def extract(self,x,c,box,reference_override=None):
        q=self.q;x=x.to(self.current.weight.dtype)
        multi=F.relu(q.front(x)) if self.kind=='quarter' else q.front(x)
        current=F.relu(self.current(x[:,-1:]));gate=self.gate(x)
        ref=q.reference(c.to(x.dtype),box,multi.shape[-2:]).to(x.dtype) if reference_override is None else reference_override.to(x.dtype)
        if self.kind=='quarter':
            base=q.head(q.body(multi))+ref if q.late_reference else q.head(q.body(multi+ref))
            if q.raw_skip:base=base+F.conv2d(x,q.raw_kernel.to(x),stride=4)
        else:
            base=q.output.conv(q.tail(q.body(multi)+ref))
            if q.raw_skip:base=base+F.pixel_unshuffle(x[:,-1:],2)
        raw_delta=F.conv2d(x,self.gate.kernel[0:1].to(x))
        current_raw=x[:,-1:];highpass=current_raw-F.avg_pool2d(current_raw,5,1,2,count_include_pad=False)
        delta_phases=F.conv2d(raw_delta,self.phase_kernel.to(x),stride=self.factor)
        highpass_phases=F.conv2d(highpass,self.phase_kernel.to(x),stride=self.factor)
        return base,torch.cat((current-multi,delta_phases,highpass_phases),1),gate

    def phases(self,x,c,box,reference_override=None):
        base,features,gate=self.extract(x,c,box,reference_override)
        return base+gate*self.residual(features)

    def native(self,x,c,box):return F.pixel_shuffle(self.phases(x,c,box),self.factor)
