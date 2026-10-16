"""Run the frozen phase model on the actual resized training inputs."""
import torch
from torch.nn import functional as F

def reference_field(reference, context):
    m=reference.board
    if not reference.kind:
        q=m.base.quarter
        return m.reference.stages(context.to(q.front.first.weight.dtype))['reference_resampled']
    s=m.source; owner=s.core.model; ref=owner.global_reference
    c=s.core.half_input(context); value=ref.encoder(c)
    if hasattr(ref,'pyramid'):
        for i in getattr(ref,'keep',range(len(ref.pyramid))):
            value=value+m.resizes[i](ref.pyramid[i](c).float()).to(value.dtype)
    value=value+value.mean((-2,-1),keepdim=True)
    if reference.kind=='ordinary':
        dtype=owner.head.weight.dtype if hasattr(owner.head,'weight') else next(owner.head.parameters()).dtype
        return ref.project(m.sample(value.float()).to(dtype))
    return m.sample(s.low_ref(ref.project(value)).float())

def native_teacher(reference, raw, field):
    raw=raw[:,:reference.source_count]
    m=reference.board
    if reference.kind:
        s=m.source; owner=s.core.model
        body=owner.body(owner.head(s.core.half_input(s.front(raw.to(s.front.temporal_conv.weight.dtype)))))
        if reference.kind=='ordinary':phases=s.output.conv(owner.tail(body+field.to(body.dtype)))
        else:phases=s.output.conv(owner.tail(body))+field.to(body.dtype)
    else:
        q=m.base.quarter; raw=raw.to(q.front.first.weight.dtype)
        body=q.body(q.front(raw))
        phases=q.output.conv(q.tail(body+field.to(body.dtype)))
        if q.raw_skip:phases=phases+F.conv2d(raw[:,-1:].to(phases.dtype),m.raw_kernel.to(phases.dtype),stride=2)
    if reference.accepted is not None:phases=phases+reference.accepted(raw)
    return F.pixel_shuffle(phases.float(),2)
