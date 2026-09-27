"""Preserve packed values, do shuffle in FP16, then display scale at output."""
import torch
from torch import nn
from torch.nn import functional as F

class HalfShuffleStudent(nn.Module):
    def __init__(self,core,student,output='float32'):
        super().__init__();self.core=core;self.student=student;self.output=output
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        joined=c.second(F.relu(c.first(c.half_input(torch.cat((stack,self.student(stack)),1))))).float()
        denoised=stack[:,-1:].float()+joined[:,:1]*(1.-torch.sigmoid(joined[:,1:]))
        features=m.body(m.head(m.down(c.half_input(denoised))))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=m.upsample[0](m.tail(features+ref))
        if self.output=='float16':return F.pixel_shuffle(packed.clamp(0,1)*255,6)
        value=F.pixel_shuffle(packed,6).float().clamp(0,1)*255
        return value.round().to(torch.uint8) if self.output=='uint8' else value
