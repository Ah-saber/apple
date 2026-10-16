"""Joint target/background residual from frozen complete observations and causal RAW."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_adaptive270 import QualityAdaptive270
from small_temporal258 import RawSmallTemporal258,features258
from large_temporal275 import RawLarge275
from small_context227 import features227
from joint_context279 import JointContext279,joint_mask279,joint_raw279
from quality_control43 import lift_native

class QualityJoint279:
    def __init__(self,number,review_only=False):
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        accepted=fit['status']=='passed' or (review_only and fit['status']=='review_only')
        model_round=fit.get('model_round',number)
        self.parent_round=283 if model_round in [284,285,286,288,292] else 273
        if not accepted or fit['current_complete_parent']!=self.parent_round:raise ValueError('Selected candidate or explicit diagnostic review required')
        self.alpha=fit['best']['alpha'];self.include_large=model_round==280;self.fixed_road=model_round in [282,283,284,285,286,288,292]
        self.source_controller=QualityJoint279(283) if self.parent_round==283 else QualityAdaptive270(273);self.trace_controller=self.source_controller.trace_controller
        if model_round in [284,285,286,288,292]:
            from tail_context284 import TailContext284
            self.model=TailContext284().cuda().eval().requires_grad_(False)
        else:self.model=JointContext279().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%model_round)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.reset_observations()
    def reset_observations(self):self.previous_body=None;self.raw_small=RawSmallTemporal258();self.raw_large=RawLarge275()
    def reset(self):self.source_controller.reset();self.reset_observations()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset_observations()
        before=self.source_controller.native(fields,row,frame,initial=initial);self.last_before_native=before.clone();self.source_native=before
        trace=self.trace_controller;whole=before[0,0].cpu().numpy();body=whole[TOP:BOTTOM]
        if self.previous_body is None:self.previous_body=body.copy()
        raw=trace.parent.last_physical_fore95.astype(np.float32)
        cues=self.source_controller.last_cues if self.parent_round==283 else joint_raw279(self.raw_small.step(raw,trace.persistent.states,trace.road),self.raw_large.step(raw,trace.persistent.states,trace.road))
        self.last_cues=cues
        proposed=before.clone()
        if trace.last_nn_context is not None:
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT);independent=trace.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous_body)/16,(body-independent)/16,(trace.last_physical_reference195-body)/16]).clip(-4,4).astype(np.float16)
            feature=features258(features227(trace.last_nn_context,extra,trace.last_release),cues)
            mask=joint_mask279(trace.last_compact_mask,trace.last_release,trace.last_active,self.include_large)
            if self.fixed_road:mask=trace.road.copy();mask[:,:64]=0
            delta=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(BOTTOM-TOP,y+128);top=max(0,y-64);bottom=min(BOTTOM-TOP,end+64)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device);weight=torch.from_numpy(mask[top:bottom])[None,None].to(before.device)
                delta[y:end]=self.model(values,weight)[0,0,y-top:end-top].cpu().numpy()
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device);proposed=proposed.clamp(0,255)
        self.previous_body=body.copy();self.last_proposed_native=proposed;self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        before=self.source_controller.full(full,self.source_native,current,shifted,fields);self.last_before_full=before
        return lift_native(before,native)
