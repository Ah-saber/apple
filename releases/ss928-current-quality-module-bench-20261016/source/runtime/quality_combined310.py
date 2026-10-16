"""Jointly selected foreground shape and background stabilization, one parent replay."""
import json,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_foreground300 import QualityForeground300
from quality_background298 import QualityBackground298
from quality_control43 import lift_native
from project_tail289 import project_tail

class QualityCombined310:
    def __init__(self,number=310):
        folder=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'
        fit=json.loads((folder/'fit.json').read_text());config=json.loads((folder/'config.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=291:raise ValueError('Joint original-train selection required')
        self.alpha=fit['best']['alpha'];self.foreground=QualityForeground300(config['foreground_model'],proposal_only=config['foreground_model'] in [312,316]);self.protect_large=config.get('protect_large',False)
        self.component_foreground=config.get('component_foreground',False);self.component_background=config.get('component_background',False)
        self.background=QualityBackground298(308,source_controller=self.foreground.source_controller);self.trace_controller=self.foreground.trace_controller
    def reset(self):self.foreground.reset();self.background.reset()
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        self.foreground.native(fields,row,frame,initial=initial)
        before=self.foreground.last_before_native;self.last_before_native=before.clone()
        self.background_native=self.background.refine(before,fields,initial=initial)
        foreground=self.foreground.last_proposed_native
        trace=self.trace_controller
        if self.component_background and trace.last_nn_context is not None:
            from component_tone320 import component_tone320
            road=trace.road.copy();road[:,:64]=0
            delta=(self.background_native-before)[0,0,TOP:BOTTOM].cpu().numpy()
            projected=component_tone320(delta,trace.last_active,road)
            self.background_native=before.clone();self.background_native[...,TOP:BOTTOM,:]+=torch.from_numpy(projected)[None,None].to(before.device)
            self.background_native=self.background_native.clamp(0,255)
        if self.protect_large and trace.last_nn_context is not None:
            change=(foreground-before)[0,0,TOP:BOTTOM].cpu().numpy();road=trace.road.copy();road[:,:64]=0
            if self.component_foreground:
                from component_tone320 import component_tone320
                projected=component_tone320(change,trace.last_active,road)
            else:projected=project_tail(change,np.zeros_like(change),trace.last_active,road,large_only=True)
            foreground=before.clone();foreground[...,TOP:BOTTOM,:]+=torch.from_numpy(projected)[None,None].to(before.device);foreground=foreground.clamp(0,255)
        proposed=(foreground+self.background_native-before).clamp(0,255)
        self.last_proposed_native=proposed;self.last_context_head=(proposed-self.trace_controller.last_base206_native)[0,0,TOP:BOTTOM].cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        background=self.background.full(full,self.background_native,current,shifted,fields)
        before=self.background.last_before_full;self.last_before_full=before
        if self.component_background and self.trace_controller.last_nn_context is not None:
            from component_tone320 import component_tone320
            image=background[0,0].cpu().numpy();parent=before[0,0].cpu().numpy();trace=self.trace_controller
            road=trace.road.copy();road[:,:64]=0
            for py in range(3):
                for px in range(3):
                    phase=image[py::3,px::3];source=parent[py::3,px::3]
                    delta=phase[TOP:BOTTOM]-source[TOP:BOTTOM]
                    phase[TOP:BOTTOM]=source[TOP:BOTTOM]+component_tone320(delta,trace.last_active,road)
            background=torch.from_numpy(image.clip(0,255))[None,None].to(before.device)
        proposed=lift_native(background,self.last_proposed_native)
        return (before+self.alpha*(proposed-before)).clamp(0,255)
