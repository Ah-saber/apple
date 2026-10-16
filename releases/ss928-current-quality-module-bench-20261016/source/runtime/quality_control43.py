"""Causal quality control; preserve the original complete three-times output."""
import json
import torch
from torch.nn import functional as F
from target_residual import ROOT
from raw_stable27 import RawStable27
from source_profile40 import source_model
from tone40 import ToneStats40,apply

def lift_native(full_gray,native_gray):
    """Anchor the correction at the existing ::3 evaluation coordinates."""
    h,w=native_gray.shape[-2:]
    if tuple(full_gray.shape[-2:])!=(3*h,3*w):raise ValueError('Complete three-times SR output required')
    original=full_gray.float()
    delta=native_gray-original[...,::3,::3]
    lifted=F.interpolate(delta,size=(3*h-2,3*w-2),mode='bilinear',align_corners=True)
    lifted=F.pad(lifted,(0,2,0,2),mode='replicate')
    return (original+lifted).clamp(0,255)

class QualityControl43:
    """Inputs are native SR variants in 0..255 gray and physical RAW features."""
    def __init__(self,scene,device='cuda'):
        self.scene=scene;self.device=device;self.previous=None;self.previous_output=None;self.memory=None
        self.guard_parameters=None;self.tone=None;self.parameters=None
        if scene=='night_special':
            fit=json.loads((ROOT/'MOTION42-20261014'/scene/'fit.json').read_text())
            if fit['status']!='passed':raise ValueError('No accepted night motion configuration')
            self.parameters=fit['best'];self.model=RawStable27()
            ck=torch.load(ROOT/'MOTION28-20261014'/scene/'last_step500.pt',map_location='cpu',weights_only=False)
            self.model.load_state_dict(ck['model']);self.source='MOTION28+CARRY42'
        else:
            self.model,self.guard_parameters,parent=source_model(scene)
            fit=json.loads((ROOT/'MOTION40-20261014'/scene/'fit.json').read_text())
            if fit['status']!='passed':raise ValueError('No accepted regional tone configuration')
            self.parameters=fit['best'];self.tone=ToneStats40(self.parameters['tiles'])
            self.source=parent+'+TONE40'
        self.model=self.model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def native(self,fields,initial=False):
        v={k:a.to(self.device,dtype=torch.float32) for k,a in fields.items() if k!='GT'}
        first=initial or self.previous is None
        previous=v if first else self.previous
        parent,motion,carry,corrected=self.model(v,previous,v['baseline'] if first else self.previous_output,initial=first)
        self.previous=v;self.previous_output=parent
        if self.scene=='night_special':
            image=corrected.clamp(0,255)
            p=self.parameters
            background=1/(1+torch.exp(((motion-p['threshold'])*20).clamp(-40,40)))
            keep=carry+(p['floor']-carry).clamp_min(0)*background
            result=image if first else (image*(1-keep)+self.memory*keep).clamp(0,255)
            self.memory=result
        else:
            image=parent[0,0].cpu().numpy().clip(0,255)
            if self.guard_parameters is not None:
                from contrast_guard18 import guard
                image=guard(image,corrected[0,0].cpu().numpy().clip(0,255),v['activity'][0,0].cpu().numpy(),self.guard_parameters)
            image=apply(image,self.tone.update(image,initial=first),self.parameters['strength'])
            result=torch.from_numpy(image)[None,None].to(self.device)
        return result

    @torch.inference_mode()
    def full(self,full_gray,fields,initial=False):
        return lift_native(full_gray,self.native(fields,initial=initial))
