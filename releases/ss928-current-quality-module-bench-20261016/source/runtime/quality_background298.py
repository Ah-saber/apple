"""Learned foreground-protected causal background stabilization on full RAW runs."""
import json
import numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_projected291 import QualityProjected291
from adaptive_static270 import AdaptiveStatic270,features270
from background_temporal295 import BackgroundTemporal295
from quality_control43 import lift_native

class QualityBackground298:
    def __init__(self,number=298,source_controller=None):
        self.number=number
        folder=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32';fit=json.loads((folder/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=291:raise ValueError('Selected current-parent background result required')
        self.alpha=fit['best']['alpha'];self.source_controller=source_controller if source_controller is not None else QualityProjected291(291);self.trace_controller=self.source_controller.trace_controller
        self.model=AdaptiveStatic270().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(folder/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.filter=BackgroundTemporal295(lowpass=1 if number in [304,308] else 9);self.previous=None
        self.full_filters=[BackgroundTemporal295(lowpass=1) for _ in range(9)];self.previous_full=None
    def reset(self):
        self.source_controller.reset();self.filter.reset();self.previous=None;self.previous_full=None
        for value in self.full_filters:value.reset()
    def infer(self,image,previous,delta,activity,occupancy,device):
        feature=features270(image,previous,delta,activity,occupancy);repair=np.zeros_like(image)
        for y in range(0,len(image),128):
            end=min(y+128,len(image));top=max(0,y-16);bottom=min(len(image),end+16)
            f=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(device);d=torch.from_numpy(delta[top:bottom].copy())[None,None].to(device)
            repair[y:end]=self.model(f,d)[0,0,y-top:end-top].cpu().numpy()
        return repair
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        before=self.source_controller.native(fields,row,frame,initial=initial)
        return self.refine(before,fields,initial=initial)
    @torch.inference_mode()
    def refine(self,before,fields,initial=False):
        if initial:
            self.filter.reset();self.previous=None;self.previous_full=None
            for value in self.full_filters:value.reset()
        self.source_native=before;self.last_before_native=before.clone()
        image=before[0,0].cpu().numpy();trace=self.trace_controller;delta=np.zeros_like(image)
        self.ready=trace.last_nn_context is not None
        if self.ready:
            road=trace.road.copy();road[:,:64]=0
            proposed=self.filter.step(image[TOP:BOTTOM],trace.last_compact_mask,trace.last_active,trace.last_release,road)
            delta[TOP:BOTTOM]=np.clip(image[TOP:BOTTOM]+proposed,0,255)-image[TOP:BOTTOM]
        else:self.filter.reset()
        if self.previous is None:self.previous=image.copy()
        self.activity=fields['activity'][0,0].cpu().numpy();self.occupancy=fields['occupancy'][0,0].cpu().numpy()
        repair=self.infer(image,self.previous,delta,self.activity,self.occupancy,before.device)
        self.previous=image.copy();proposed=(before+torch.from_numpy(repair)[None,None].to(before.device)).clamp(0,255)
        self.last_proposed_native=proposed;self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        before=self.source_controller.full(full,self.source_native,current,shifted,fields);self.last_before_full=before
        if self.number in [304,308]:
            image=before[0,0].cpu().numpy();candidate=image.copy();trace=self.trace_controller
            if self.previous_full is None:self.previous_full=image.copy()
            road=trace.road.copy();road[:,:64]=0
            for py in range(3):
                for px in range(3):
                    phase=image[py::3,px::3];delta=np.zeros_like(phase)
                    filter=self.full_filters[3*py+px]
                    if self.ready:
                        raw=filter.step(phase[TOP:BOTTOM],trace.last_compact_mask,trace.last_active,trace.last_release,road)
                        delta[TOP:BOTTOM]=np.clip(phase[TOP:BOTTOM]+raw,0,255)-phase[TOP:BOTTOM]
                    else:filter.reset()
                    repair=self.infer(phase,self.previous_full[py::3,px::3],delta,self.activity,self.occupancy,before.device)
                    candidate[py::3,px::3]+=self.alpha*repair
            self.previous_full=image.copy()
            return torch.from_numpy(candidate.clip(0,255))[None,None].to(before.device)
        return lift_native(before,native)
