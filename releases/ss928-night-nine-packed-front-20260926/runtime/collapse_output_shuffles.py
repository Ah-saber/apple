"""Fold consecutive pixel shuffles into a single one by fixed phase permutation."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def collapse_output_shuffles(model):
    source=model.upsample
    factors=[part.upscale_factor for part in list(source)[1:]]
    if len(factors)<2:
        return model
    scale=math.prod(factors)
    phases=torch.arange(scale*scale).reshape(1,scale*scale,1,1)
    for factor in factors:
        phases=F.pixel_shuffle(phases,factor)
    order=phases.flatten().long().to(source[0].weight.device)
    conv=source[0]
    with torch.no_grad():
        conv.weight.copy_(conv.weight.index_select(0,order))
        if conv.bias is not None:
            conv.bias.copy_(conv.bias.index_select(0,order))
    model.upsample=nn.Sequential(conv,nn.PixelShuffle(scale))
    return model
