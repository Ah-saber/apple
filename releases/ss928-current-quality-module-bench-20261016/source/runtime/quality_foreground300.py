"""Correct foreground shape with larger range inside causal RAW ownership."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_projected291 import QualityProjected291
from foreground_shape300 import ForegroundShape300,owner300,features300
from small_context227 import features227
from small_temporal258 import features258
from quality_control43 import lift_native

class QualityForeground300:
    def __init__(self,number=300,proposal_only=False):
        self.number=number
        folder=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32';fit=json.loads((folder/'fit.json').read_text())
        parent=311 if number in [318,326,328] else 291
        accepted=fit['status']=='passed' or (proposal_only and fit['status']=='no_pass')
        if not accepted or fit['current_complete_parent']!=parent:raise ValueError('Selected foreground or explicit proposal for joint selection required')
        self.alpha=fit['best']['alpha']
        if number in [318,326,328]:
            from quality_combined310 import QualityCombined310
            self.source_controller=QualityCombined310(311)
        else:self.source_controller=QualityProjected291(291)
        self.trace_controller=self.source_controller.trace_controller
        if number==316:
            from raw_amplitude316 import ForegroundAmplitude316,RawAmplitude316
            self.model=ForegroundAmplitude316().cuda().eval().requires_grad_(False);self.raw_amplitude=RawAmplitude316()
        else:self.model=ForegroundShape300().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(folder/'last_step20000.pt',map_location='cpu',weights_only=False)['model']);self.previous=None
    def reset(self):
        self.source_controller.reset();self.previous=None
        if self.number==316:self.raw_amplitude.history.clear()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:
            self.previous=None
            if self.number==316:self.raw_amplitude.history.clear()
        before=self.source_controller.native(fields,row,frame,initial=initial);self.source_native=before;self.last_before_native=before.clone()
        whole=before[0,0].cpu().numpy();body=whole[TOP:BOTTOM];trace=self.trace_controller
        if self.number==316:
            amplitude=self.raw_amplitude.step(trace.parent.last_physical_fore95.astype(np.float32),trace.persistent.states,trace.road)
        if self.previous is None:self.previous=body.copy()
        proposed=before.clone()
        if trace.last_nn_context is not None:
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT)
            independent=trace.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous)/16,(body-independent)/16,(trace.last_physical_reference195-body)/16]).clip(-4,4).astype(np.float16)
            cues=self.source_controller.foreground.source_controller.last_cues if self.number in [318,326,328] else self.source_controller.last_cues
            feature=features300(features258(features227(trace.last_nn_context,extra,trace.last_release),cues),owner300(trace.last_compact_mask,trace.last_active))
            if self.number==316:
                from raw_amplitude316 import features316
                feature=features316(feature,amplitude)
            mask=trace.road.copy();mask[:,:64]=0;delta=np.zeros_like(body)
            if self.number in [326,328]:
                from foreground_support324 import foreground_support324
                mask*=foreground_support324(trace.last_compact_mask,trace.last_active)
            for y in range(0,len(body),128):
                end=min(y+128,len(body));top=max(0,y-64);bottom=min(len(body),end+64)
                f=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device);m=torch.from_numpy(mask[top:bottom])[None,None].to(before.device)
                delta[y:end]=self.model(f,m)[0,0,y-top:end-top].cpu().numpy()
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device);proposed=proposed.clamp(0,255)
        self.previous=body.copy();self.last_proposed_native=proposed
        self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        before=self.source_controller.full(full,self.source_native,current,shifted,fields);self.last_before_full=before
        return lift_native(before,native)
