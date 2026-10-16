"""Refine the target branch above the current complete result using the same RAW observations."""
import json
import numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_static259 import QualityStatic259
from small_temporal258 import SmallTemporal258,features258
from quality_control43 import lift_native

class QualityTarget271:
    def __init__(self,number=271):
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=263:raise ValueError('Original train selection relative to current complete result required')
        self.alpha=fit['best']['alpha'];self.source_controller=QualityStatic259(263);self.trace_controller=self.source_controller.trace_controller
        self.target=json.loads((ROOT/('MOTION%d-20261015'%number)/'branches.json').read_text())['target']
        target_fit=json.loads((ROOT/('MOTION%d-20261015'%self.target)/'weather_heavy_c32'/'fit.json').read_text());self.target_alpha=target_fit['best']['alpha']
        self.model=SmallTemporal258().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%self.target)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.reset_raw()
    def reset_raw(self):
        self.raw_observations=None
        if self.target==269:
            from raw_aligned269 import RawAligned269
            self.raw_observations=RawAligned269()
    def reset(self):self.source_controller.reset();self.reset_raw()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset_raw()
        before=self.source_controller.native(fields,row,frame,initial=initial);trace=self.trace_controller;old_branch=self.source_controller.source_controller
        self.last_before_native=before.clone();self.source_native=before
        cues=None
        if self.raw_observations is not None:cues=self.raw_observations.step(trace.parent.last_physical_fore95.astype(np.float32),trace.persistent.states,trace.road)
        proposed=before.clone()
        if old_branch.last_feature is not None:
            feature=old_branch.last_feature if cues is None else features258(old_branch.last_feature[:19],cues)
            delta=np.zeros((BOTTOM-TOP,1280),np.float32)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-64);bottom=min(BOTTOM-TOP,end+64)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device);mask=torch.from_numpy(trace.last_small_mask[top:bottom])[None,None].to(before.device)
                delta[y:end]=self.model(values,mask)[0,0,y-top:end-top].cpu().numpy()
            base=old_branch.last_before_native
            raw=base.clone();raw[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device);raw=raw.clamp(0,255)
            new=base+self.target_alpha*(raw-base)
            proposed=(before+new-self.source_controller.source_native).clamp(0,255)
        self.last_proposed_native=proposed;self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        before=self.source_controller.full(full,self.source_native,current,shifted,fields);self.last_before_full=before
        return lift_native(before,native)
