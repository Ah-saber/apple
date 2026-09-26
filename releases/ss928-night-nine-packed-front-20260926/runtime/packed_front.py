"""Learn denoised packed2 tensor directly; keep the frozen display backbone."""
import torch
from torch import nn
from torch.nn import functional as F

class PackedFront(nn.Module):
    def __init__(self,width=12):
        super().__init__()
        self.stem=nn.Conv2d(10,8,6,stride=2,padding=2)
        self.down=nn.Conv2d(8,width,3,stride=2,padding=1)
        self.body=nn.ModuleList([nn.Conv2d(width,width,5,padding=2) for _ in range(4)])
        self.output=nn.ConvTranspose2d(width,4,4,stride=2,padding=1)
        self.skip=nn.Conv2d(8,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.skip.weight);nn.init.zeros_(self.skip.bias)
    def forward(self,stack):
        current=stack[:,-1:].float()
        residual=(stack.float()-stack[:,:6].float().mean(1,keepdim=True))*64.
        raw=torch.cat((residual,current),1).to(dtype=self.stem.weight.dtype,memory_format=torch.channels_last)
        half=F.relu(self.stem(raw));x=F.relu(self.down(half))
        for layer in self.body:x=x+F.relu(layer(x))*.25
        correction=(self.output(x)+self.skip(half)).float()*.025
        return F.pixel_unshuffle(current,2)+correction

class PackedFullNine(nn.Module):
    def __init__(self,core,front,output='float32',shuffle_half=True):
        super().__init__();self.core=core;self.front=front;self.output=output;self.shuffle_half=shuffle_half
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        denoised=self.front(stack)
        features=m.body(m.head(c.half_input(denoised)))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=m.upsample[0](m.tail(features+ref))
        if self.output=='float16':
            return F.pixel_shuffle(packed.clamp(0,1)*255,6)
        if self.shuffle_half:
            gray=F.pixel_shuffle(packed,6).float().clamp(0,1)*255
            return gray.round().to(torch.uint8) if self.output=='uint8' else gray
        gray=packed.float().clamp(0,1)*255
        return F.pixel_shuffle(gray.round().to(torch.uint8) if self.output=='uint8' else gray,6)
