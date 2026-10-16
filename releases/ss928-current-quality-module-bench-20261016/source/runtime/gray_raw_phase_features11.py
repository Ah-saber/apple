"""Use gray-unit RAW features; fold the scale into fitted inference kernels."""
import torch
from torch.nn import functional as F
from current_raw_phase_features11 import expand,unshuffle
def features(branch,current,clean):
    actual=unshuffle(current);denoised=unshuffle(clean)
    source=torch.cat([actual-denoised,actual-F.avg_pool2d(actual,3,1,1)],1)
    conv=branch.current_features
    if getattr(branch,'current_features_folded',False):return conv(source)
    return F.conv2d(source,conv.weight*255,conv.bias,padding=1,groups=8)
def fold(branch):
    if getattr(branch,'current_features_folded',False):raise ValueError('RAW feature scale already folded')
    with torch.no_grad():branch.current_features.weight.mul_(255)
    branch.current_features_folded=True
