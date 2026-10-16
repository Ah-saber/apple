"""Causal compact-target correction and larger-body background stabilization."""
import json
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from body_ownership167 import owner167
from quality_small227 import QualitySmall227
from motion_background226 import MotionBackground226

class QualityCombined231(QualitySmall227):
    def __init__(self,number=231):
        super().__init__(236 if number==237 else 230)
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=222:raise ValueError('Combined original-train selection required')
        self.combined_alpha=fit['best']['alpha']
        self.large_alpha=json.loads((ROOT/'MOTION228-20261015'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']

    def reset(self):
        super().reset();self.background_stabilizer=MotionBackground226()

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        proposed=super().native(fields,row,frame,initial=initial)
        before=self.last_before_native
        if self.last_nn_context is not None:
            body=before[0,0,TOP:BOTTOM].float().cpu().numpy()
            physical=self.parent.last_physical_fore95.astype(np.float32)
            owner=owner167(self.persistent.states,physical)
            confirmed=np.maximum(np.maximum(owner,self.tracks.last[1]),self.last_protect)
            independent=self.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            delta,_=self.background_stabilizer.step(body,independent,physical,confirmed,self.last_support_mask,self.road,owner,self.last_protect)
            n,labels,boxes,_=cv2.connectedComponentsWithStats((self.last_active>0).astype(np.uint8),connectivity=8)
            selected=np.zeros_like(labels,dtype=np.uint8)
            for k in range(1,n):
                if boxes[k,cv2.CC_STAT_AREA]>=128:selected[labels==k]=1
            distance=cv2.distanceTransform((selected==0).astype(np.uint8),cv2.DIST_C,3)
            route=np.clip((24-distance)/12,0,1)
            increment=self.large_alpha*route*(np.clip(body+delta,0,255)-body)
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(increment)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-self.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.combined_alpha*(proposed-before)).clamp(0,255)
