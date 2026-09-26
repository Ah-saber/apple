"""Exact phase permutation and fixed-select pack controls for complete models."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def phase_permutation(first,second):
    r=first*second
    # CRD: after first shuffle C=second**2. Final offsets are
    # (second*i+p, second*j+q), where c=p*second+q.
    return [(second*i+p)*r+second*j+q
            for p in range(second) for q in range(second)
            for i in range(first) for j in range(first)]


class FixedCurrentPack(nn.Module):
    """FP32 fixed convolution selects last RAW frame's four pixel phases."""
    def __init__(self,front):
        super().__init__();self.front=front
        weight=torch.zeros(4,9,2,2,device=front.stem.weight.device,dtype=torch.float32)
        for dy in range(2):
            for dx in range(2):weight[dy*2+dx,8,dy,dx]=1.
        self.register_buffer('pack_weight',weight)
    def forward(self,stack):
        f=self.front;current=stack[:,-1:].float()
        residual=(stack.float()-stack[:,:6].float().mean(1,keepdim=True))*64.
        raw=torch.cat((residual,current),1).to(dtype=f.stem.weight.dtype,memory_format=torch.channels_last)
        half=F.relu(f.stem(raw));x=F.relu(f.down(half))
        for layer in f.body:x=x+F.relu(layer(x))*.25
        correction=(f.output(x)+f.skip(half)).float()*.025
        packed=F.conv2d(stack.float(),self.pack_weight,stride=2)
        return packed+correction


class CompleteVariant(nn.Module):
    def __init__(self,base,first=6,second=1,fixed_pack=False,output='float32'):
        super().__init__()
        if first*second!=6:raise ValueError('Need total shuffle scale six')
        self.core=base.core;self.front=FixedCurrentPack(base.front) if fixed_pack else base.front
        self.first,self.second,self.output=first,second,output
        self.output_conv=copy.deepcopy(base.core.model.upsample[0])
        if second!=1:
            order=phase_permutation(first,second)
            with torch.no_grad():
                self.output_conv.weight.copy_(self.output_conv.weight[order].clone())
                if self.output_conv.bias is not None:self.output_conv.bias.copy_(self.output_conv.bias[order].clone())
        self.eval()
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        features=m.body(m.head(c.half_input(self.front(stack))))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=self.output_conv(m.tail(features+ref))
        if self.output=='float16':packed=packed.clamp(0,1)*255
        gray=F.pixel_shuffle(packed,self.first)
        if self.second!=1:gray=F.pixel_shuffle(gray,self.second)
        if self.output=='float16':return gray
        gray=gray.float().clamp(0,1)*255
        return gray.round().to(torch.uint8) if self.output=='uint8' else gray


class MeanRawDisplay(nn.Module):
    """Average the nine displayed subpixels per RAW pixel, then replicate in model."""
    def __init__(self,base,two_stage=False,output='float32'):
        super().__init__();self.core=base.core;self.front=base.front;self.two_stage=two_stage;self.output=output
        old=base.core.model.upsample[0]
        self.output_conv=nn.Conv2d(16,4,3,padding=1,bias=old.bias is not None).to(device=old.weight.device,dtype=old.weight.dtype,memory_format=torch.channels_last)
        with torch.no_grad():
            for dy in range(2):
                for dx in range(2):
                    indices=[(3*dy+p)*6+3*dx+q for p in range(3) for q in range(3)]
                    self.output_conv.weight[dy*2+dx].copy_(old.weight[indices].float().mean(0))
                    if old.bias is not None:self.output_conv.bias[dy*2+dx].copy_(old.bias[indices].float().mean(0))
        weight=torch.zeros(4,1,6,6,device=old.weight.device,dtype=old.weight.dtype)
        for dy in range(2):
            for dx in range(2):weight[dy*2+dx,0,dy*3:dy*3+3,dx*3:dx*3+3]=1
        self.register_buffer('repeat6_weight',weight)
        self.register_buffer('repeat3_weight',torch.ones(1,1,3,3,device=old.weight.device,dtype=old.weight.dtype))
        self.eval()
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        features=m.body(m.head(c.half_input(self.front(stack))))
        ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=self.output_conv(m.tail(features+ref))
        if self.output=='float16':packed=packed.clamp(0,1)*255
        if self.two_stage:
            gray=F.conv_transpose2d(F.pixel_shuffle(packed,2),self.repeat3_weight,stride=3)
        else:gray=F.conv_transpose2d(packed,self.repeat6_weight,stride=6)
        if self.output=='float16':return gray
        gray=gray.float().clamp(0,1)*255
        return gray.round().to(torch.uint8) if self.output=='uint8' else gray


class MeanRawDitherDisplay(MeanRawDisplay):
    """Preserve sub-gray RAW averages after uint8 rounding with fixed phase offsets."""
    def __init__(self,base,two_stage=False,output='float32'):
        super().__init__(base,two_stage,output)
        device=self.output_conv.weight.device
        offsets=torch.arange(-4,5,device=device,dtype=torch.float32).reshape(3,3)/9.
        weight6=torch.zeros(5,1,6,6,device=device,dtype=torch.float32)
        weight6[:4]=self.repeat6_weight.float()
        for dy in range(2):
            for dx in range(2):weight6[4,0,dy*3:dy*3+3,dx*3:dx*3+3]=offsets
        weight3=torch.ones(2,1,3,3,device=device,dtype=torch.float32);weight3[1,0]=offsets
        self.register_buffer('dither6_weight',weight6);self.register_buffer('dither3_weight',weight3)
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        features=m.body(m.head(c.half_input(self.front(stack))))
        ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=self.output_conv(m.tail(features+ref)).float().clamp(0,1)*255
        if self.two_stage:
            raw=F.pixel_shuffle(packed,2)
            one=torch.ones((stack.shape[0],1,stack.shape[-2],stack.shape[-1]),device=stack.device,dtype=torch.float32)
            gray=F.conv_transpose2d(torch.cat((raw,one),1),self.dither3_weight,stride=3)
        else:
            one=torch.ones((stack.shape[0],1,stack.shape[-2]//2,stack.shape[-1]//2),device=stack.device,dtype=torch.float32)
            gray=F.conv_transpose2d(torch.cat((packed,one),1),self.dither6_weight,stride=6)
        gray=gray.clamp(0,255)
        if self.output=='float16':return gray.half()
        return gray.round().to(torch.uint8) if self.output=='uint8' else gray
