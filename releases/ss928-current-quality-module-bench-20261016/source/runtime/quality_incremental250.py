"""Causal residual refinement above the retained complete small-target/surround result."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_combined231 import QualityCombined231
from small_residual249 import SmallResidual249
from small_context227 import features227
from learned_background243 import LearnedBackground243,build_masks243
from body_ownership167 import owner167

def selection(number):
    fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
    if fit['status']!='passed' or fit['current_complete_parent']!=237:raise ValueError('Original-train selection relative to retained parent required')
    return fit['best']['alpha']

class QualityIncremental250(QualityCombined231):
    def __init__(self,number):
        super().__init__(237)
        self.incremental_number=number;self.incremental_alpha=selection(number)
        branch_numbers=[247,251] if number==252 else [247,249] if number==250 else [number]
        self.incremental_branches={n:selection(n) for n in branch_numbers}
        self.incremental_models={}
        for n in self.incremental_branches:
            model=LearnedBackground243() if n==247 else SmallResidual249()
            checkpoint_number=249 if n==251 else n
            model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%checkpoint_number)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
            self.incremental_models[n]=model.cuda().eval().requires_grad_(False)

    def reset(self):
        super().reset();self.previous_retained_base=None

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        before=super().native(fields,row,frame,initial=initial)
        self.last_before_native=before.clone()
        whole=before[0,0].float().cpu().numpy();body=whole[TOP:BOTTOM]
        if self.previous_retained_base is None:self.previous_retained_base=body.copy()
        proposed=before.clone()
        if self.last_nn_context is not None:
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT)
            independent=self.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous_retained_base)/16,
                            (body-independent)/16,(self.last_physical_reference195-body)/16]).clip(-4,4).astype(np.float16)
            feature=features227(self.last_nn_context,extra,self.last_release)
            increment=np.zeros_like(body)
            for n,model in self.incremental_models.items():
                if n==247:
                    raw=self.parent.last_physical_fore95.astype(np.float32)
                    owner=owner167(self.persistent.states,raw)
                    masks=build_masks243(body,raw,owner,self.tracks.last[1],self.last_protect,self.last_active,self.last_support_mask,self.road,whole_owner=True)
                else:masks=self.last_small_mask[None]
                delta=np.zeros_like(body)
                for y in range(0,BOTTOM-TOP,128):
                    end=min(y+128,BOTTOM-TOP);top=max(0,y-64);bottom=min(BOTTOM-TOP,end+64)
                    values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                    local_masks=torch.from_numpy(masks[:,top:bottom])[None].to(before.device)
                    value=model.background(values,local_masks) if n==247 else model(values,local_masks)
                    delta[y:end]=value[0,0,y-top:end-top].cpu().numpy()
                if n==247:
                    delta=model.project(torch.from_numpy(delta)[None,None].to(before.device),torch.from_numpy(masks)[None].to(before.device))[0,0].cpu().numpy()
                branch=np.clip(body+delta,0,255)-body
                increment+=(self.incremental_branches[n] if self.incremental_number in [250,252] else 1.)*branch
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(increment)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.previous_retained_base=body.copy()
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-self.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.incremental_alpha*(proposed-before)).clamp(0,255)
