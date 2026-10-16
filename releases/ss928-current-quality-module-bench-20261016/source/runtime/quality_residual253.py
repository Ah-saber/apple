"""Residual inference above a frozen complete controller, without output feedback."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_incremental250 import QualityIncremental250
from small_residual249 import SmallResidual249
from small_context227 import features227

class QualityResidual253:
    def __init__(self,number):
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=252:raise ValueError('Original-train selection relative to frozen complete parent required')
        self.alpha=fit['best']['alpha']
        self.source_controller=QualityIncremental250(252)
        self.trace_controller=self.source_controller
        self.model=SmallResidual249().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.previous_source_body=None

    def reset(self):
        self.source_controller.reset();self.previous_source_body=None

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.previous_source_body=None
        parent=self.source_controller
        before=parent.native(fields,row,frame,initial=initial)
        self.last_before_native=before.clone()
        whole=before[0,0].float().cpu().numpy();body=whole[TOP:BOTTOM]
        if self.previous_source_body is None:self.previous_source_body=body.copy()
        proposed=before.clone()
        if parent.last_nn_context is not None:
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT)
            independent=parent.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous_source_body)/16,
                            (body-independent)/16,(parent.last_physical_reference195-body)/16]).clip(-4,4).astype(np.float16)
            feature=features227(parent.last_nn_context,extra,parent.last_release)
            delta=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-64);bottom=min(BOTTOM-TOP,end+64)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                masks=torch.from_numpy(parent.last_small_mask[top:bottom])[None,None].to(before.device)
                delta[y:end]=self.model(values,masks)[0,0,y-top:end-top].cpu().numpy()
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.previous_source_body=body.copy()
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-parent.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)

    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        self.source_controller.last_before_native=self.last_before_native
        result=self.source_controller.full(full,native,current,shifted,fields)
        self.last_before_full=self.source_controller.last_before_full
        return result
