"""Run compact physical104 online and stabilize physical edge differences over3 frames."""
import json
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from target_residual import ROOT
from quality_c3292 import QualityC3292
from matched_positive104 import MatchedPositive104
from physical_reference90 import FrozenTone90,detail,clean_common
from fixed_background98 import features98
from compact_match93 import TOP,BOTTOM
from road_residual53 import support53
from edge_reference91 import WIDTH91,PAD91,mask91

class StableEdge106:
    def __init__(self):self.native_bank=deque(maxlen=2);self.full_bank=deque(maxlen=2)
    def native(self,fields):
        difference=fields['shifted_current91']-fields['current_only'][...,:WIDTH91]
        observation=difference[...,:3].clone()
        median=torch.quantile(torch.stack(list(self.native_bank)+[observation]),.5,dim=0)
        active=fields['activity'][...,:3].clamp(0,1)
        selected=active*observation+(1-active)*median
        self.native_bank.append(observation)
        self.edge=torch.clamp(difference,-8,8)*torch.from_numpy(mask91()).to(difference.device)
        self.edge[...,:3]=selected
        return selected-torch.clamp(observation,-8,8)
    def full(self,full,native,current,shifted,fields):
        h,w=native.shape[-2:]
        observation=shifted[...,3*PAD91:3*PAD91+9]-current[...,:9]
        median=torch.quantile(torch.stack(list(self.full_bank)+[observation]),.5,dim=0)
        active=F.interpolate(fields['activity'][...,:3].clamp(0,1),scale_factor=3,mode='nearest')
        selected=active*observation+(1-active)*median
        self.full_bank.append(observation.clone())
        delta=native-full[...,::3,::3]
        lifted=F.interpolate(delta,size=(3*h-2,3*w-2),mode='bilinear',align_corners=True)
        quality=full+F.pad(lifted,(0,2,0,2),mode='replicate')
        dense=F.interpolate(self.edge,size=(3*h-2,3*WIDTH91-2),mode='bilinear',align_corners=True)
        dense=F.pad(dense,(0,2,0,2),mode='replicate')
        quality[...,:9]+=selected-dense[...,:9]
        quality[...,::3,:9:3]=native[...,:,:3]
        return quality.clamp(0,255)

class QualityC32106:
    def __init__(self,number=106):
        if number!=106:raise ValueError('This controller is round106')
        self.parent=QualityC3292(92)
        self.gain=json.loads((ROOT/'MOTION95-20261014'/'weather_heavy_c32'/'response.json').read_text())['gain']
        self.coefficients=np.asarray(json.loads((ROOT/'MOTION98-20261014'/'weather_heavy_c32'/'background.json').read_text())['coefficients'],np.float32)
        fit98=json.loads((ROOT/'MOTION98-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        fit104=json.loads((ROOT/'MOTION104-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        if fit98['status']!='passed' or fit104['status']!='passed':raise ValueError('Original training selections required')
        self.alpha98=fit98['best']['alpha'];self.alpha104=fit104['best']['alpha']
        self.road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
        self.center=self.road>.9;self.center[:,:64]=False
        self.source='Original SR3 -> train-selected physical compact104 -> three-frame physical complete edge106'
        self.reset()
    def reset(self):
        self.bank=deque(maxlen=64);self.tone=None;self.reference=None;self.fixed=None
        self.matcher=MatchedPositive104(self.road);self.edge=StableEdge106()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset()
        source=self.parent.native(fields,row,frame,initial=initial)
        image=source[0,0].float().cpu().numpy().copy();independent=image.copy()
        current_sr=image[TOP:BOTTOM].copy();raw_controller=self.parent.parent.parent
        values=raw_controller.last_body_fields
        gate=values['signed_gate80'][0,0].float().cpu().numpy()
        presence=values['presence76'][0,0].float().cpu().numpy()
        physical_fore=np.zeros_like(current_sr);fade=0.
        background=np.median(np.stack(self.bank),axis=0) if self.bank else current_sr
        foreground=current_sr-background;foreground-=np.median(foreground[self.center])
        if len(self.bank)>=16:
            physical=raw_controller.raw.open[row['raw']['path']]
            if self.tone is None:
                initial_raw=np.median(np.asarray(physical[frame-16:frame],np.float32),axis=0)
                self.tone=FrozenTone90(initial_raw);self.reference=initial_raw[TOP:BOTTOM]
                fixed_physical=self.tone.apply(detail(self.reference))
                self.fixed=np.clip(np.einsum('c,cyx->yx',self.coefficients,features98(fixed_physical,background)),-6,6)*self.road
            raw_background=np.median(np.asarray(physical[max(0,frame-64):frame,TOP:BOTTOM],np.float32),axis=0)
            raw_background=clean_common(raw_background,self.reference)
            current=clean_common(physical[frame,TOP:BOTTOM].astype(np.float32),raw_background)
            physical_fore=self.tone.apply(detail(current))-self.tone.apply(detail(raw_background))
            physical_fore-=np.median(physical_fore[self.center])
            fade=min((len(self.bank)-15)/3,1)
        proposal=self.matcher.step(presence,gate,foreground,physical_fore,self.gain)
        self.last_source92=independent
        self.last_physical_fore95=physical_fore.copy()
        self.last_body_delta104=np.zeros_like(current_sr)
        self.last_positive104=np.zeros_like(current_sr)
        self.last_base104=current_sr.copy()
        self.last_fixed_background98=np.zeros_like(current_sr)
        if fade:
            self.last_fixed_background98=self.alpha104*self.alpha98*self.fixed*fade
            self.last_body_delta104=self.alpha104*(self.alpha98*self.fixed*fade+proposal)
            self.last_positive104=self.alpha104*proposal
            self.last_base104=np.clip(current_sr+self.alpha104*self.alpha98*self.fixed*fade,0,255)
            image[TOP:BOTTOM]+=self.last_body_delta104
        image=image.clip(0,255)
        self.last_unguarded104=image[TOP:BOTTOM].copy();self.last_gate104=gate
        if hasattr(self.edge,'set_motion'):self.edge.set_motion(gate)
        self.last_edge_extra106=self.edge.native(fields)[0,0].float().cpu().numpy()
        image[:,:3]+=self.last_edge_extra106
        image=image.clip(0,255)
        self.bank.append(current_sr)
        self.last_delta106=image-independent
        return torch.from_numpy(image)[None,None].cuda()
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        return self.edge.full(full,native,current,shifted,fields)
