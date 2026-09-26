"""Learned nine-frame trajectory surrogate; frozen downstream network is retained."""
import types
import torch
from torch import nn
from torch.nn import functional as F

class TrajectoryStudent(nn.Module):
    def __init__(self,width=12,quarter_stem=False):
        super().__init__()
        self.quarter_stem=quarter_stem
        self.stem=nn.Conv2d(9,width if quarter_stem else 8,5,stride=4 if quarter_stem else 2,padding=2)
        self.down=nn.Identity() if quarter_stem else nn.Conv2d(8,width,3,stride=2,padding=1)
        self.body=nn.ModuleList([nn.Conv2d(width,width,5,padding=2) for _ in range(4)])
        self.output=nn.ConvTranspose2d(width,4,8,stride=4,padding=2)
        self.register_buffer('limits',torch.tensor([10.,20.,10.,20.]).reshape(1,4,1,1))
        nn.init.constant_(self.output.bias,-2.)
    def forward(self,stack):
        # This one temporal mean retains FP32 subtraction before FP16 convolutions.
        residual=(stack.float()-stack[:,:6].float().mean(1,keepdim=True))*64.
        x=residual.to(dtype=self.stem.weight.dtype,memory_format=torch.channels_last)
        x=F.relu(self.stem(x));x=F.relu(self.down(x))
        for layer in self.body:x=x+F.relu(layer(x))*.25
        return torch.sigmoid(self.output(x).float())*self.limits.float()

def student_features(self,stack):return self.trajectory_student(stack)

def attach_student(core,student):
    core.model.trajectory_student=student
    core.model._trajectory_features=types.MethodType(student_features,core.model)
    return core

class StudentFullNine(nn.Module):
    """Explicit new forward avoids reuse of dynamically patched module-method graphs."""
    def __init__(self,core,student):
        super().__init__();self.core=core;self.student=student
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        gate_input=torch.cat((stack,self.student(stack)),1)
        joined=c.second(F.relu(c.first(c.half_input(gate_input)))).float()
        denoised=stack[:,-1:].float()+joined[:,:1]*(1.-torch.sigmoid(joined[:,1:]))
        features=m.body(m.head(m.down(c.half_input(denoised))))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=m.upsample[0](m.tail(features+ref)).float().clamp(0,1)*255
        return F.pixel_shuffle(packed,6)

class StudentByteNine(StudentFullNine):
    """Match previous GPU route: FP16 RAW input and complete uint8 output."""
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        gate_input=torch.cat((stack,self.student(stack)),1)
        joined=c.second(F.relu(c.first(c.half_input(gate_input)))).float()
        denoised=stack[:,-1:].float()+joined[:,:1]*(1.-torch.sigmoid(joined[:,1:]))
        features=m.body(m.head(m.down(c.half_input(denoised))))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=m.upsample[0](m.tail(features+ref)).float().clamp(0,1)*255
        return F.pixel_shuffle(packed.round().to(torch.uint8),6)
