"""Shared temporal/motion convolution graph, with optional raw-basis transform."""
import torch
from torch import nn
from torch.nn import functional as F

class FusedNine(nn.Module):
    def __init__(self, model, raw_basis=False, channels_last=False, dense_second=False):
        super().__init__()
        self.raw_basis=raw_basis
        self.channels_last=channels_last
        temporal,motion=model.temporal_pre,model.motion_pre
        first=nn.Conv2d(13,16,5,padding=2)
        second=nn.Conv2d(16,2,5,padding=2,groups=1 if dense_second else 2)
        with torch.no_grad():
            first.weight.zero_()
            first.weight[:8,:9].copy_(temporal[0].weight)
            first.weight[8:].copy_(motion[0].weight)
            first.bias.copy_(torch.cat((temporal[0].bias,motion[0].bias)))
            if dense_second:
                second.weight.zero_()
                second.weight[:1,:8].copy_(temporal[2].weight)
                second.weight[1:,8:].copy_(motion[2].weight)
            else:second.weight.copy_(torch.cat((temporal[2].weight,motion[2].weight)))
            second.bias.copy_(torch.cat((temporal[2].bias,motion[2].bias)))
            if raw_basis:
                weight=first.weight[:,:9].clone()
                first.weight[:,:8].copy_(weight[:,1:9].flip(1))
                first.weight[:,8].copy_(weight[:,0]-weight[:,1:9].sum(1))
        self.first=first.half()
        self.second=second.half()
        self.model=model.half()
        if channels_last:self.to(memory_format=torch.channels_last)
        self.factors=[p.upscale_factor for p in list(model.upsample)[1:]]

    def half_input(self,x):
        if self.channels_last:return x.to(dtype=torch.float16,memory_format=torch.channels_last)
        return x.half()

    def forward(self,stack,context):
        m=self.model
        current=stack[:,-1:]
        if self.raw_basis:input9=stack
        else:input9=torch.cat((current,*(stack[:,j:j+1]-current for j in range(7,-1,-1))),1)
        gate_input=torch.cat((input9,m._trajectory_features(stack)),1)
        joined=self.second(F.relu(self.first(self.half_input(gate_input)))).float()
        correction=joined[:,:1]
        gate=torch.sigmoid(joined[:,1:])
        denoised=current.float()+correction*(1.-gate)
        features=m.body(m.head(m.down(self.half_input(denoised))))
        reference=m.global_reference.encode(self.half_input(context))
        reference=m.global_reference.project(reference+reference.mean((-2,-1),keepdim=True))
        reference=F.interpolate(reference,size=features.shape[-2:],mode='bilinear',align_corners=False)
        packed=m.upsample[0](m.tail(features+reference))
        display=torch.round(packed.float().clamp(0,1)*255).to(torch.uint8)
        for factor in self.factors:display=F.pixel_shuffle(display,factor)
        return display
