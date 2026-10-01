"""Current RAW phases with bounded coarse-history features and small residual model."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class ReliableModel(nn.Module):
    factor=4
    def __init__(self,reference,width=8,threshold_gray=2.):
        super().__init__()
        self.threshold_gray=threshold_gray
        self.reference=copy.deepcopy(reference)
        old=self.reference.project
        self.reference.project=nn.Conv2d(old.in_channels,1,1)
        nn.init.zeros_(self.reference.project.weight);nn.init.zeros_(self.reference.project.bias)
        self.front=nn.Conv2d(18,width,1)
        self.body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU())
        self.head=nn.Conv2d(width,16,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        kernel=torch.zeros(16,1,4,4)
        for i in range(16):kernel[i,0,i//4,i%4]=1
        self.register_buffer('current_kernel',kernel)

    def components(self,stack):
        dtype=self.front.weight.dtype
        stack=stack.to(dtype)
        current=F.conv2d(stack[:,-1:],self.current_kernel.to(dtype),stride=4)
        means=F.avg_pool2d(stack,4,4)
        current_mean=means[:,-1:]
        difference=means[:,:8]-current_mean
        threshold=self.threshold_gray/255.
        reliability=(1-difference.abs()/threshold).clamp(0,1)
        # Each weighted difference is bounded by threshold/4. This bound applies
        # to the history feature, not to the learned residual or final image.
        contribution=(reliability*difference).mean(1,keepdim=True)
        return current,current_mean,contribution,reliability

    def phases(self,stack,context,box):
        current,mean,history,reliability=self.components(stack)
        features=F.relu(self.front(torch.cat((current,mean,history),1)))
        learned=self.head(self.body(features))
        reference=self.reference(context.to(features.dtype),box,current.shape[-2:]).to(features.dtype)
        return current+learned+reference

    def native(self,stack,context,box):return F.pixel_shuffle(self.phases(stack,context,box),4)

    def forward(self,stack,context,box):
        return F.interpolate(self.native(stack,context,box).clamp(0,1)*255,scale_factor=3,mode='nearest')
