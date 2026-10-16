"""Apply adaptive background strength to each phase of complete original-input output."""
import json
import numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_static259 import QualityStatic259
from adaptive_static270 import AdaptiveStatic270,features270

class QualityAdaptive270:
    def __init__(self,number=270):
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=263:raise ValueError('Original train selection relative to current complete result required')
        self.alpha=fit['best']['alpha'];self.combined=number==273
        if self.combined:
            from quality_target271 import QualityTarget271
            self.source_controller=QualityTarget271(271);self.static_controller=self.source_controller.source_controller
            self.static_alpha=json.loads((ROOT/'MOTION270-20261015'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
        else:self.source_controller=QualityStatic259(263);self.static_controller=self.source_controller;self.static_alpha=1.
        self.trace_controller=self.static_controller.trace_controller
        self.model=AdaptiveStatic270().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/'MOTION270-20261015'/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.previous_native=None;self.previous_full=None
    def reset(self):self.source_controller.reset();self.previous_native=None;self.previous_full=None
    def infer(self,image,previous,delta,activity,occupancy,device):
        feature=features270(image,previous,delta,activity,occupancy);repair=np.zeros_like(image)
        for y in range(0,len(image),128):
            end=min(y+128,len(image));top=max(0,y-16);bottom=min(len(image),end+16)
            f=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(device);d=torch.from_numpy(delta[top:bottom].copy())[None,None].to(device)
            repair[y:end]=self.model(f,d)[0,0,y-top:end-top].cpu().numpy()
        return repair
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.previous_native=None;self.previous_full=None
        source=self.source_controller.native(fields,row,frame,initial=initial)
        before=self.source_controller.last_before_native if self.combined else source
        self.last_before_native=before.clone();self.source_native=source
        image=before[0,0].cpu().numpy()
        if self.previous_native is None:self.previous_native=image.copy()
        self.activity=fields['activity'][0,0].cpu().numpy();self.occupancy=fields['occupancy'][0,0].cpu().numpy()
        repair=self.infer(image,self.previous_native,self.static_controller.last_static_delta,self.activity,self.occupancy,source.device)
        self.previous_native=image.copy()
        proposed=(source+self.static_alpha*torch.from_numpy(repair)[None,None].to(source.device)).clamp(0,255)
        self.last_proposed_native=proposed;self.last_context_head=(proposed-self.trace_controller.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        source=self.source_controller.full(full,self.source_native,current,shifted,fields)
        before=self.source_controller.last_before_full if self.combined else source
        image=before[0,0].cpu().numpy();candidate=source[0,0].cpu().numpy().copy()
        if self.previous_full is None:self.previous_full=image.copy()
        raw_delta=self.static_controller.last_full_static_delta
        for py in range(3):
            for px in range(3):
                repair=self.infer(image[py::3,px::3],self.previous_full[py::3,px::3],raw_delta[py::3,px::3],self.activity,self.occupancy,source.device)
                candidate[py::3,px::3]+=self.static_alpha*repair
        self.previous_full=image.copy();self.last_before_full=before
        proposed=torch.from_numpy(candidate.clip(0,255))[None,None].to(source.device)
        return (before+self.alpha*(proposed-before)).clamp(0,255)
