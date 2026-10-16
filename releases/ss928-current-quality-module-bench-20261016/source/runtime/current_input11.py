"""Fold repeated-current temporal kernels; keep three causal frames for detail."""
import torch
from torch import nn

def folded_conv(source,count):
    if not isinstance(source,nn.Conv2d) or source.groups!=1:raise ValueError('Dense temporal input convolution required')
    new=nn.Conv2d(count,source.out_channels,source.kernel_size,stride=source.stride,padding=source.padding,dilation=source.dilation,bias=source.bias is not None,device=source.weight.device,dtype=source.weight.dtype)
    with torch.no_grad():
        if count==1:new.weight.copy_(source.weight.sum(1,keepdim=True))
        else:
            new.weight.copy_(source.weight[:,-count:])
            dropped=source.weight[:,:source.in_channels-count].sum(1,keepdim=True)
            new.weight[:,:count-1].add_(dropped/(count-1))
        if source.bias is not None:new.bias.copy_(source.bias)
    return new

class CurrentInput(nn.Module):
    def __init__(self,source):super().__init__();self.source=source
    @property
    def temporal_conv(self):return self.source.temporal_conv
    @property
    def first(self):return self.source.first
    @property
    def conv(self):return self.source.conv
    def forward(self,raw):return self.source(raw[:,-1:])

def fold_current(model):
    reference=model.reference
    if reference.kind:
        front=reference.board.source.front
        old_count=front.temporal_conv.in_channels
        front.temporal_conv=folded_conv(front.temporal_conv,1)
        reference.board.source.front=CurrentInput(front)
    else:
        front=reference.board.base.quarter.front
        old_count=front.first.in_channels
        front.first=folded_conv(front.first,1)
        reference.board.base.quarter.front=CurrentInput(front)
    if reference.accepted is not None:
        count=reference.accepted.conv.in_channels
        def visit(parent):
            for name,child in list(parent.named_children()):
                if isinstance(child,nn.Conv2d) and child.in_channels==count:setattr(parent,name,folded_conv(child,1))
                else:visit(child)
        visit(reference.accepted);reference.accepted=CurrentInput(reference.accepted)
    model.branch.encode=folded_conv(model.branch.encode,3)
    model.branch.count=3
    return dict(main_current_frames=1,detail_causal_frames=3,old_front_frames=old_count,temporal_DC_preserved=True,changing_frame_function_preserved=False)
