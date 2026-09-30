"""Weight-preserving day/half-grid graph variants; NPU mapping remains unmeasured."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from compact_model import FixedSample


class FrozenReference(nn.Module):
    def __init__(self, reference, box, height, width, projection_after=False):
        super().__init__()
        self.reference = copy.deepcopy(reference)
        self.projection_after = projection_after
        self.sample = FixedSample(box,height,width)
        self.resizes = nn.ModuleList([FixedSample(torch.tensor([[0.,0.,1.,1.]]),64,64,64//s)
                                     for s in (2,4,8)])

    def encode(self, image):
        ref=self.reference
        x=ref.encoder(image)
        if hasattr(ref,'pyramid'):
            for i,branch in enumerate(ref.pyramid):
                x=x+self.resizes[i](branch(image).float()).to(x.dtype)
        return x

    def forward(self, image):
        return self.stages(image)['reference_resampled']

    def stages(self, image):
        stages={}
        x=self.reference.encoder(image)
        stages['reference_encoder']=x
        if hasattr(self.reference,'pyramid'):
            for i,branch in enumerate(self.reference.pyramid):
                part=branch(image)
                stages['reference_pyramid_'+str(i)]=part
                x=x+self.resizes[i](part.float()).to(x.dtype)
        stages['reference_encoded']=x
        x=x+x.mean((-2,-1),keepdim=True)
        stages['reference_mean_added']=x
        if self.projection_after:
            result=self.reference.project(self.sample(x.float()).to(x.dtype))
        else:
            projected=self.reference.project(x)
            stages['reference_projected']=projected
            result=self.sample(projected.float())
        stages['reference_resampled']=result
        return stages


class BoardHalfOptimized(nn.Module):
    def __init__(self, base, box, layout='rows32', single_frame=False,
                 projection_after=False, native_height=1024,native_width=1280):
        super().__init__()
        from export_half_board import Rows32
        self.base=copy.deepcopy(base)
        self.single_frame=single_frame
        q=self.base.quarter
        if single_frame:
            assert self.base.current_only, 'Only fold a model already trained with repeated current frames'
            old=q.front.first
            first=nn.Conv2d(1,old.out_channels,old.kernel_size,old.stride,old.padding)
            with torch.no_grad():
                first.weight.copy_(old.weight.sum(1,keepdim=True))
                first.bias.copy_(old.bias)
            q.front.first=first
        self.reference=FrozenReference(q.reference,box,native_height//2,native_width//2,projection_after)
        self.rows=Rows32()
        self.layout=layout
        kernel=torch.zeros(4,1,2,2)
        for i in range(4): kernel[i,0,i//2,i%2]=1
        self.register_buffer('raw_kernel',kernel)

    def stages(self, stack, context):
        q=self.base.quarter
        if self.base.current_only and not self.single_frame:
            stack=stack[:,-1:].expand_as(stack)
        front=q.front(stack)
        body=q.body(front)
        reference_stages=self.reference.stages(context.to(q.front.first.weight.dtype))
        reference=reference_stages['reference_resampled'].to(body.dtype)
        fused=body+reference
        tail=q.tail(fused)
        learned=q.output.conv(tail)
        phases=learned
        raw_phases=None
        if q.raw_skip:
            raw_phases=F.conv2d(stack[:,-1:].to(learned.dtype),self.raw_kernel.to(learned.dtype),stride=2)
            phases=phases+raw_phases
        stages={'front':front,'body':body,'reference':reference,'fused':fused,
                'tail':tail,'learned_phases':learned,'preclip_phases':phases}
        if raw_phases is not None: stages['raw_phases']=raw_phases
        stages.update(reference_stages)
        return stages

    def forward(self, stack, context):
        phases=self.stages(stack,context)['preclip_phases'].clamp(0,1)*255
        if self.layout=='phases': return phases
        if self.layout=='byte':
            return self.rows(phases).to(torch.uint8)
        return self.rows(phases)
