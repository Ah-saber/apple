"""Apply static-detail fusion independently to all nine SR phases, with RAW target guards."""
import json
import numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_residual253 import QualityResidual253
from body_ownership167 import owner167
from static_highpass259 import StaticHighpass259,motion259

class QualityStatic259:
    def __init__(self,number):
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=254:raise ValueError('Original-train selection relative to frozen parent required')
        self.alpha=fit['best']['alpha'];self.combined=number==263
        if self.combined:
            from quality_temporal258 import QualityTemporal258
            self.source_controller=QualityTemporal258()
            config=json.loads((ROOT/'MOTION263-20261015'/'branches.json').read_text());self.variant=config['static'];self.static_alpha=config['strengths'][str(self.variant)]
        else:self.source_controller=QualityResidual253(254);self.variant=number;self.static_alpha=1.
        self.trace_controller=self.source_controller.trace_controller
        self.reset_filters()

    def reset_filters(self):
        options=dict(preserve_peaks=self.variant==261,preserve_compact=self.variant==262,protected_columns=3 if self.variant==264 else 64)
        self.native_filter=StaticHighpass259(**options);self.full_filters=[StaticHighpass259(**options) for _ in range(9)]

    def reset(self):self.source_controller.reset();self.reset_filters()

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset_filters()
        source=self.source_controller.native(fields,row,frame,initial=initial);trace=self.trace_controller
        before=self.source_controller.last_before_native.clone() if self.combined else source.clone()
        self.last_before_native=before;self.source_native=source
        image=source[0,0].float().cpu().numpy();raw=trace.parent.last_physical_fore95.astype(np.float32)
        owner=owner167(trace.persistent.states,raw)
        protect=trace.last_protect if trace.last_nn_context is not None else trace.tracks.last_protection
        activity=fields['activity'][0,0].float().cpu().numpy();occupancy=fields['occupancy'][0,0].float().cpu().numpy()
        self.current_motion=motion259(activity,occupancy,owner,trace.tracks.last[1],protect,
            binary=self.variant in [261,262],raw_gate=trace.parent.last_gate104 if self.variant==264 else None)
        delta=self.native_filter.step(image,self.current_motion)
        self.last_static_delta=np.clip(image+delta,0,255)-image
        proposed_image=image+self.static_alpha*(np.clip(image+delta,0,255)-image)
        proposed=torch.from_numpy(proposed_image.clip(0,255))[None,None].to(source.device)
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-trace.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.alpha*(proposed-before)).clamp(0,255)

    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        source=self.source_controller.full(full,self.source_native,current,shifted,fields)
        before=self.source_controller.last_before_full if self.combined else source
        image=source[0,0].float().cpu().numpy();proposed=image.copy();raw_delta=np.zeros_like(image)
        for py in range(3):
            for px in range(3):
                phase=image[py::3,px::3]
                delta=self.full_filters[3*py+px].step(phase,self.current_motion)
                raw_delta[py::3,px::3]=delta
                proposed[py::3,px::3]=phase+self.static_alpha*(np.clip(phase+delta,0,255)-phase)
        candidate=torch.from_numpy(proposed)[None,None].to(source.device)
        self.last_full_static_delta=np.clip(image+raw_delta,0,255)-image
        self.last_before_full=before
        return (before+self.alpha*(candidate-before)).clamp(0,255)
