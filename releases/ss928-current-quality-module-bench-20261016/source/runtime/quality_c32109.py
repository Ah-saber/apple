"""Median independent108 observations in the static original91 boundary band."""
from collections import deque
import torch
from torch.nn import functional as F
from quality_c32108 import QualityC32108
from compact_match93 import TOP,BOTTOM
from edge_reference91 import WIDTH91,mask91

class StaticBand109:
    def __init__(self):
        self.native_bank=deque(maxlen=2);self.full_bank=deque(maxlen=2)
        self.motion_bank=deque(maxlen=2)
    def native_observation(self,independent):return independent[...,:WIDTH91].clone()
    def full_observation(self,independent):return independent[...,:3*WIDTH91].clone()
    def native(self,independent,fields,gate,road):
        observed=self.native_observation(independent)
        motion=fields['activity'][...,:WIDTH91+1].clamp(0,1).clone()
        signed=torch.from_numpy(gate[:,:WIDTH91+1].copy()).to(observed.device)[None,None].abs().div(.15).clamp(0,1)
        onroad=torch.from_numpy((road[:,:WIDTH91+1]>.1).copy()).to(observed.device)[None,None]
        motion[...,TOP:BOTTOM,:]=torch.where(onroad,signed,motion[...,TOP:BOTTOM,:])
        union=torch.stack(list(self.motion_bank)+[motion]).amax(dim=0)
        active=F.max_pool2d(union,3,stride=1,padding=1)[...,:WIDTH91]
        mask=torch.from_numpy(mask91()).to(observed.device)[None,None]
        self.weight=mask*(1-active)
        median=torch.quantile(torch.stack(list(self.native_bank)+[observed]),.5,dim=0)
        result=independent.clone();result[...,:WIDTH91]+=self.weight*(median-observed)
        self.native_bank.append(observed);self.motion_bank.append(motion)
        return result
    def full(self,independent):
        observed=self.full_observation(independent)
        median=torch.quantile(torch.stack(list(self.full_bank)+[observed]),.5,dim=0)
        weight=F.interpolate(self.weight,scale_factor=3,mode='nearest')
        result=independent.clone();result[...,:3*WIDTH91]+=weight*(median-observed)
        self.full_bank.append(observed)
        return result.clamp(0,255)

class QualityC32109(QualityC32108):
    def __init__(self,number=109):
        if number!=109:raise ValueError('This controller is round109')
        super().__init__(108)
        self.source='Original SR3 -> trained107 and frozen physical edge108 -> static current+past2 complete edge band109'
    def reset(self):
        super().reset();self.static_band=StaticBand109()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        independent=super().native(fields,row,frame,initial=initial)
        self.independent_native108=independent
        return self.static_band.native(independent,fields,self.last_gate104,self.road)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        independent=super().full(full,self.independent_native108,current,shifted,fields)
        return self.static_band.full(independent)
