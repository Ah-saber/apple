"""Causal feature blending: trained spatial/reference paths are retained."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class HistorySignal(nn.Module):
    def __init__(self,threshold,factor,force=None):
        super().__init__();self.factor=factor;self.force=force
        # Current minus eight-frame mean and current minus nearest history.
        kernel=torch.zeros(2,9,1,1);kernel[:,8]=1;kernel[0,:8]=-1/8;kernel[1,7]=-1
        self.register_buffer('kernel',kernel);self.register_buffer('threshold',torch.tensor(float(threshold)))

    def signal(self,x):
        difference=F.conv2d(x,self.kernel.to(x))
        smooth=F.avg_pool2d(difference,3,1,1,count_include_pad=False)
        return torch.maximum(smooth[:,0:1].abs(),smooth[:,1:2].abs())

    def forward(self,x):
        if self.force is not None:
            return torch.ones_like(x[:,-1:,::self.factor,::self.factor])*self.force
        signal=self.signal(x)
        tau=self.threshold.to(x)
        gate=((signal-tau)/tau.clamp_min(1e-6)).clamp(0,1)
        return F.max_pool2d(gate,self.factor,self.factor) if self.factor>1 else gate


def fold_current(conv):
    result=nn.Conv2d(1,conv.out_channels,conv.kernel_size,stride=conv.stride,padding=conv.padding,
                     bias=conv.bias is not None,device=conv.weight.device,dtype=conv.weight.dtype)
    with torch.no_grad():
        result.weight.copy_(conv.weight.sum(1,keepdim=True))
        if conv.bias is not None:result.bias.copy_(conv.bias)
    return result


class AdaptiveHistory(nn.Module):
    def __init__(self,source,kind,threshold,force=None):
        super().__init__();self.source=copy.deepcopy(source);self.kind=kind;self.factor=4 if kind=='quarter' else 2
        self.current=fold_current(source.front if kind=='quarter' else source.quarter.front.first)
        self.gate=HistorySignal(threshold,self.factor,force)

    @property
    def q(self):return self.source if self.kind=='quarter' else self.source.quarter

    def phases(self,x,c,box,reference_override=None):
        q=self.q;x=x.to(self.current.weight.dtype)
        multi=F.relu(q.front(x)) if self.kind=='quarter' else q.front(x)
        current=F.relu(self.current(x[:,-1:]));gate=self.gate(x)
        features=multi+gate*(current-multi)
        ref=q.reference(c.to(x.dtype),box,features.shape[-2:]).to(x.dtype) if reference_override is None else reference_override.to(x.dtype)
        if self.kind=='quarter':
            phases=q.head(q.body(features))+ref if q.late_reference else q.head(q.body(features+ref))
            if q.raw_skip:phases=phases+F.conv2d(x,q.raw_kernel.to(x),stride=4)
        else:
            phases=q.output.conv(q.tail(q.body(features)+ref))
            if q.raw_skip:phases=phases+F.pixel_unshuffle(x[:,-1:],2)
        return phases

    def native(self,x,c,box):return F.pixel_shuffle(self.phases(x,c,box),self.factor)


class GatedNightFront(nn.Module):
    def __init__(self,source,threshold,force=None,strength=1.):
        super().__init__();self.source=copy.deepcopy(source);self.current=fold_current(source.temporal_conv)
        self.gate=HistorySignal(threshold,1,force);self.strength=float(strength)
        # Expose the dtype contract used by the preserved input preparation function.
        self.first=self.source.first

    def forward(self,x):
        s=self.source;x=x.to(dtype=s.first.weight.dtype,memory_format=torch.channels_last)
        multi=F.relu(s.temporal_conv(x));current=F.relu(self.current(x[:,-1:]));gate=self.gate(x)
        value=multi+self.strength*gate*(current-multi)
        return s.last(F.relu(s.first(value.to(memory_format=torch.channels_last))))
