"""Select a RAW-owned foreground redistribution without advancing the parent twice."""
import json,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_foreground300 import QualityForeground300
from project_shape330 import project_shape330
from quality_control43 import lift_native

class QualityShape330:
    def __init__(self,number):
        folder=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'
        fit=json.loads((folder/'fit.json').read_text());self.config=json.loads((folder/'config.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=311:raise ValueError('Selected foreground redistribution required')
        self.alpha=fit['best']['alpha'];self.foreground=QualityForeground300(326);self.trace_controller=self.foreground.trace_controller
    def reset(self):self.foreground.reset()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        self.foreground.native(fields,row,frame,initial=initial)
        before=self.foreground.last_before_native;self.last_before_native=before.clone();proposed=before.clone();trace=self.trace_controller
        if trace.last_nn_context is not None:
            delta=(self.foreground.last_proposed_native-before)[0,0,TOP:BOTTOM].cpu().numpy()
            road=trace.road.copy();road[:,:64]=0
            delta=project_shape330(delta,trace.last_compact_mask,trace.last_active,road,small_only=self.config['small_only'],conserve_mean=self.config['mean_preserving'])
            delta[:,:64]=0;proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(delta)[None,None].to(before.device)
        proposed=proposed.clamp(0,255);self.last_proposed_native=proposed
        self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        before=self.foreground.source_controller.full(full,self.foreground.source_native,current,shifted,fields)
        self.last_before_full=before
        return lift_native(before,native)
