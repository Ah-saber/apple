"""Use trained107 foreground and a frozen past16 physical edge calibration."""
import json
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from target_residual import ROOT
from quality_c32106 import QualityC32106
from peak_guard107 import PeakGuard107
from compact_match93 import TOP,BOTTOM
from edge_reference91 import WIDTH91,PAD91,mask91

class FrozenEdge108:
    def __init__(self):
        self.native_bank=deque(maxlen=16);self.full_bank=deque(maxlen=16)
        self.native_fixed=None;self.full_fixed=None
    def set_motion(self,gate):
        self.motion=torch.from_numpy(gate.copy())[None,None].abs()
        self.motion=F.max_pool2d(self.motion,3,stride=1,padding=1)[...,:3].div(.5).clamp(0,1)
    def native(self,fields):
        difference=fields['shifted_current91']-fields['current_only'][...,:WIDTH91]
        observation=difference[...,:3].clone()
        if self.native_fixed is None and len(self.native_bank)==16:
            self.native_fixed=torch.quantile(torch.stack(list(self.native_bank)),.5,dim=0)
        if self.native_fixed is None:
            reference=torch.quantile(torch.stack(list(self.native_bank)+[observation]),.5,dim=0)
            self.native_bank.append(observation)
        else:reference=self.native_fixed
        self.active=torch.zeros_like(observation)
        self.active[...,TOP:BOTTOM,:]=self.motion.to(observation.device)
        selected=self.active*observation+(1-self.active)*reference
        self.edge=torch.clamp(difference,-8,8)*torch.from_numpy(mask91()).to(difference.device)
        self.edge[...,:3]=selected
        return selected-torch.clamp(observation,-8,8)
    def full(self,full,native,current,shifted,fields):
        h,w=native.shape[-2:]
        observation=shifted[...,3*PAD91:3*PAD91+9]-current[...,:9]
        if self.full_fixed is None and len(self.full_bank)==16:
            self.full_fixed=torch.quantile(torch.stack(list(self.full_bank)),.5,dim=0)
        if self.full_fixed is None:
            reference=torch.quantile(torch.stack(list(self.full_bank)+[observation]),.5,dim=0)
            self.full_bank.append(observation.clone())
        else:reference=self.full_fixed
        active=F.interpolate(self.active,scale_factor=3,mode='nearest')
        selected=active*observation+(1-active)*reference
        delta=native-full[...,::3,::3]
        lifted=F.interpolate(delta,size=(3*h-2,3*w-2),mode='bilinear',align_corners=True)
        quality=full+F.pad(lifted,(0,2,0,2),mode='replicate')
        dense=F.interpolate(self.edge,size=(3*h-2,3*WIDTH91-2),mode='bilinear',align_corners=True)
        dense=F.pad(dense,(0,2,0,2),mode='replicate')
        quality[...,:9]+=selected-dense[...,:9]
        quality[...,::3,:9:3]=native[...,:,:3]
        return quality.clamp(0,255)

class QualityC32108(QualityC32106):
    def __init__(self,number=108):
        if number!=108:raise ValueError('This controller is round108')
        super().__init__(106)
        fit=json.loads((ROOT/'MOTION107-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed':raise ValueError('Peak107 training selection required')
        self.alpha107=fit['best']['alpha']
        self.source='Original SR3 -> train-selected compact107 with matched contrast ceiling -> frozen past16 complete edge108'
    def reset(self):
        super().reset();self.edge=FrozenEdge108();self.guard=PeakGuard107(self.road)
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        super().native(fields,row,frame,initial=initial)
        reduction=self.guard.step(self.last_unguarded104,self.last_base104,self.last_positive104,
            self.matcher.last_matches,self.last_gate104)
        image=self.last_source92.copy()
        image[TOP:BOTTOM]+=self.alpha107*(self.last_body_delta104-reduction)
        image=image.clip(0,255);self.last_body_image107=image.copy()
        image[:,:3]+=self.last_edge_extra106
        image=image.clip(0,255)
        self.last_delta108=image-self.last_source92
        return torch.from_numpy(image)[None,None].cuda()
