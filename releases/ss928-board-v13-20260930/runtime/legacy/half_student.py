"""Half-grid alternative for preserving day scene structure at low latency."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class HalfFront(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.first=nn.Conv2d(9,width,4,stride=2,padding=1)

    def forward(self,x):
        return F.relu(self.first(x.to(dtype=self.first.weight.dtype,
                    memory_format=torch.channels_last)))


class HalfSystem(nn.Module):
    factor=2

    def __init__(self,reference,depth=1,width=16,raw_skip=False):
        super().__init__()
        self.raw_skip=raw_skip
        self.front=HalfFront(width)
        layers=[]
        for _ in range(depth):
            layers.extend((nn.Conv2d(width,width,3,padding=1),nn.ReLU()))
        self.body=nn.Sequential(*layers)
        self.reference=copy.deepcopy(reference)
        old=self.reference.project
        self.reference.project=nn.Conv2d(old.in_channels,width,
            old.kernel_size,padding=old.padding)
        self.tail=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU())
        self.output=nn.Module()
        self.output.conv=nn.Conv2d(width,4,3,padding=1)
        if raw_skip:
            nn.init.zeros_(self.output.conv.weight)
            nn.init.zeros_(self.output.conv.bias)


def make_student(reference,checkpoint):
    if checkpoint.get('grid','quarter')=='half':
        return HalfSystem(reference,depth=checkpoint.get('depth',1),
                          width=checkpoint.get('width',16),
                          raw_skip=checkpoint.get('raw_skip',False))
    from quarter_candidates import QuarterSystem
    return QuarterSystem(reference,depth=checkpoint.get('depth',1),
                         mode='pixel',front_kind=checkpoint.get('front_kind','k4'),
                         width=checkpoint.get('width',32))
