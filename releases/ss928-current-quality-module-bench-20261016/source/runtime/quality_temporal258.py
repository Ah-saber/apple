"""Current small-target residual using aligned causal RAW observations."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_residual253 import QualityResidual253
from small_context227 import features227
from small_temporal258 import SmallTemporal258,RawSmallTemporal258,features258

class QualityTemporal258:
    def __init__(self):
        fit=json.loads((ROOT/'MOTION258-20261015'/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=254:raise ValueError('Original-train selection relative to frozen parent required')
        self.alpha=fit['best']['alpha'];self.source_controller=QualityResidual253(254)
        self.trace_controller=self.source_controller.trace_controller
        self.model=SmallTemporal258().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/'MOTION258-20261015'/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.previous_source_body=None;self.raw_observations=RawSmallTemporal258()

    def reset(self):
        self.source_controller.reset();self.previous_source_body=None;self.raw_observations=RawSmallTemporal258()

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        self.last_feature=None
        if initial:self.previous_source_body=None;self.raw_observations=RawSmallTemporal258()
        before=self.source_controller.native(fields,row,frame,initial=initial);trace=self.trace_controller
        self.last_before_native=before.clone()
        whole=before[0,0].float().cpu().numpy();body=whole[TOP:BOTTOM]
        if self.previous_source_body is None:self.previous_source_body=body.copy()
        raw=trace.parent.last_physical_fore95.astype(np.float32)
        cues=self.raw_observations.step(raw,trace.persistent.states,trace.road)
        proposed=before.clone()
        if trace.last_nn_context is not None:
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT)
            independent=trace.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous_source_body)/16,
                            (body-independent)/16,(trace.last_physical_reference195-body)/16]).clip(-4,4).astype(np.float16)
            feature=features258(features227(trace.last_nn_context,extra,trace.last_release),cues)
            self.last_feature=feature
            delta=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-64);bottom=min(BOTTOM-TOP,end+64)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                masks=torch.from_numpy(trace.last_small_mask[top:bottom])[None,None].to(before.device)
                delta[y:end]=self.model(values,masks)[0,0,y-top:end-top].cpu().numpy()
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.previous_source_body=body.copy();self.last_proposed_native=proposed
        self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)

    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        self.source_controller.last_before_native=self.last_before_native
        result=self.source_controller.full(full,native,current,shifted,fields)
        self.last_before_full=self.source_controller.last_before_full
        return result
