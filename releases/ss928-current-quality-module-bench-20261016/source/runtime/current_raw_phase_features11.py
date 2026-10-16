"""Expose current RAW phase contrast to the display head, preserving the clean RAW path."""
import torch
from torch import nn
from torch.nn import functional as F
def unshuffle(raw):
    n,channels,height,width=raw.shape
    if channels!=1:raise ValueError('Current RAW must have one channel')
    return raw.reshape(n,height//2,2,width//2,2).permute(0,2,4,1,3).reshape(n,4,height//2,width//2)
def expand(branch):
    source=branch.display_decode
    if source.in_channels!=16:raise ValueError('Fixed16-channel latent parent required')
    kw=dict(device=source.weight.device,dtype=source.weight.dtype)
    wider=nn.Conv2d(24,4,1,**kw);detail=nn.Conv2d(8,8,3,padding=1,groups=8,**kw)
    with torch.no_grad():
        wider.weight.zero_();wider.weight[:,:16].copy_(source.weight);wider.bias.copy_(source.bias)
        detail.weight.zero_();detail.weight[:,:,1,1]=1;detail.bias.zero_()
    branch.display_decode=wider;branch.current_features=detail
    return dict(parent_display_features=16,display_features=24,extra_current_RAW_features=8,extra_current_depthwise_kernel=3,new_display_weights_zero=True)
def features(branch,current,clean):
    actual=unshuffle(current);denoised=unshuffle(clean)
    source=torch.cat([actual-denoised,actual-F.avg_pool2d(actual,3,1,1)],1)
    return branch.current_features(source)
