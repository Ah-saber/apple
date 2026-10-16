"""Stabilize independent108 boundary low frequencies, retaining current HP9."""
import torch
from torch.nn import functional as F
from quality_c32109 import QualityC32109,StaticBand109
from edge_reference91 import WIDTH91

class StaticLowpass112(StaticBand109):
    def native_observation(self,independent):
        value=independent[...,:WIDTH91+4]
        return F.avg_pool2d(F.pad(value,(4,4,4,4),mode='reflect'),9,stride=1)[...,:WIDTH91].clone()
    def full_observation(self,independent):
        phases=F.pixel_unshuffle(independent[...,:3*(WIDTH91+4)],3)
        lowpass=F.avg_pool2d(F.pad(phases,(4,4,4,4),mode='reflect'),9,stride=1)
        return F.pixel_shuffle(lowpass[...,:WIDTH91],3).clone()

class QualityC32112(QualityC32109):
    def __init__(self,number=112):
        if number!=112:raise ValueError('This controller is round112')
        super().__init__(109)
        self.source='Original SR3 -> trained107 frozen physical edge108 -> static independent108 LP9 current+past2 boundary median112'
    def reset(self):
        super().reset();self.static_band=StaticLowpass112()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        return super().native(fields,row,frame,initial=initial).clamp(0,255)
