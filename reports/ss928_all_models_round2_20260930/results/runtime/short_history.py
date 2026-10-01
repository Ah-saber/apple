"""Retain trained spatial/reference paths while shortening the temporal first convolution."""
import copy
import torch
from torch import nn


class ShortHistory(nn.Module):
    def __init__(self,source,count,kind,current_bias=.75):
        super().__init__();assert count in [1,3,9]
        self.source=copy.deepcopy(source);self.count=count;self.kind=kind
        old=self.source.front if kind=='quarter' else self.source.quarter.front.first
        new=nn.Conv2d(count,old.out_channels,old.kernel_size,stride=old.stride,padding=old.padding,
                      bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            if count==1:weights=old.weight.sum(1,keepdim=True)
            elif count==3:weights=old.weight.reshape(old.out_channels,3,3,*old.kernel_size).sum(2)
            else:
                # Bias toward current-frame response without assuming old weights are reliable.
                weights=old.weight.clone();past=weights[:,:-1].sum(1);weights[:,:-1]*=1-current_bias;weights[:,-1]+=current_bias*past
            new.weight.copy_(weights)
            if old.bias is not None:new.bias.copy_(old.bias)
        if kind=='quarter':
            self.source.front=new
            if count<9:self.source.raw_kernel=self.source.raw_kernel[:,-count:].clone()
        else:self.source.quarter.front.first=new;self.source.current_only=False

    @property
    def reference(self):return self.source.reference if self.kind=='quarter' else self.source.quarter.reference

    def native(self,stack,context,box):
        return self.source.native(stack[:,-self.count:],context,box)

    def forward(self,stack,context,box):
        return self.source(stack[:,-self.count:],context,box)
