"""Apply selected tail cleanup while limiting current larger-foreground changes."""
import json
import numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_joint279 import QualityJoint279
from quality_control43 import lift_native
from project_tail289 import project_tail

class QualityProjected291:
    def __init__(self,number):
        folder=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'
        fit=json.loads((folder/'fit.json').read_text());config=json.loads((folder/'config.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=283:raise ValueError('Selected projection with complete parent required')
        self.alpha=fit['best']['alpha'];self.source_controller=QualityJoint279(config['model_round']);self.trace_controller=self.source_controller.trace_controller
    def reset(self):self.source_controller.reset()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        selected=self.source_controller.native(fields,row,frame,initial=initial)
        self.last_cues=self.source_controller.last_cues
        before=self.source_controller.last_before_native;self.last_before_native=before.clone()
        proposed=before.clone();trace=self.trace_controller
        if trace.last_nn_context is not None:
            delta=(selected-before)[0,0,TOP:BOTTOM].cpu().numpy()
            road=trace.road.copy();road[:,:64]=0
            projected=project_tail(delta,trace.last_compact_mask,trace.last_active,road,large_only=True)
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(projected)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        parent=self.source_controller.source_controller
        before=parent.full(full,self.last_before_native,current,shifted,fields)
        self.last_before_full=before
        return lift_native(before,native)
