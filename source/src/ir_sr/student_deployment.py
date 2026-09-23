"""Static whole-frame graph. Input uses the versioned frame/segment normalization."""
import torch
from torch import nn
from torch.nn import functional as F
from ir_sr.model import to_deploy

class FullFrameStudent(nn.Module):
    """RAW-only inference: thumbnail and full-frame reference alignment inside graph.

    Fixed 1024x1280 normalized input. Does not implement offline calibration or
    training ROI/crop behavior. Train-time crop sampling remains in the trainer.
    """
    def __init__(self, model, hierarchical_pooling=False):
        super().__init__()
        self.model = to_deploy(model)
        self.thumbnail_pool = nn.Sequential(nn.AvgPool2d((4,4)),nn.AvgPool2d((4,5))) if hierarchical_pooling else nn.AvgPool2d((16,20))
        self.reference_pool = nn.Sequential(nn.AvgPool2d(8),nn.AvgPool2d(8)) if hierarchical_pooling else nn.AvgPool2d(64)
        self.reference_resize = nn.Upsample(size=(1024//model.packing_factor,1280//model.packing_factor),mode='bilinear',align_corners=False)
        if not hasattr(self.model,'global_reference'):
            raise ValueError('Expected a trained RAW reference branch')

    def forward(self, raw):
        if not torch.jit.is_tracing() and tuple(raw.shape) != (1,1,1024,1280):
            raise ValueError('Static full-frame model requires 1x1x1024x1280')
        m=self.model
        features=m.body(m.head(m.down(raw)))
        context=self.thumbnail_pool(raw)
        ref=m.global_reference.encoder(context)
        ref=m.global_reference.project(ref+self.reference_pool(ref))
        ref=self.reference_resize(ref)
        display=m.upsample(m.tail(features+ref.to(features.dtype)))
        if m.raw_skip:
            display=display.float()+F.interpolate(raw.float(),size=(3072,3840),mode='bilinear',align_corners=False)
        return display
