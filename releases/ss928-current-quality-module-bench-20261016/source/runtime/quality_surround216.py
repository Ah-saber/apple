"""Original-input causal context refinement, with protected target contrast bands."""
import json
from collections import deque
import cv2,numpy as np,torch
from torch import nn
from target_residual import ROOT
from quality_c32206 import QualityC32206
from compact_match93 import TOP,BOTTOM
from body_ownership167 import protection167,owner167
from halo_weight163 import grow163
from small_route180 import route180
from surround_stable211 import SurroundStable211
from context_refine213 import ContextRefine213,assemble213
from quality_control43 import lift_native
from reclaim221 import reclaim221

class CaptureStep:
    def __init__(self,inner):self.inner=inner;self.last=None;self.previous_histories={}
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self,*args,**kwargs):
        if hasattr(self.inner,'states'):
            self.previous_histories={id(s):list(s['history']) for s in self.inner.states}
        self.last=self.inner.step(*args,**kwargs)
        return self.last

class CaptureFeatures(nn.Module):
    def __init__(self,inner):super().__init__();self.inner=inner;self.records=[]
    def forward(self,features,proposal):
        self.records.append(features[0].float().cpu().numpy())
        return self.inner(features,proposal)

class QualitySurround216(QualityC32206):
    def __init__(self,number=216):
        super().__init__(206)
        self.number=number
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed':raise ValueError('Original train selection required')
        self.context_alpha=fit['best']['alpha']
        self.before_alpha=json.loads((ROOT/'MOTION216-20261015'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
        self.context_model=ContextRefine213().cuda().eval().requires_grad_(False)
        checkpoint=torch.load(ROOT/'MOTION215-20261015'/'weather_heavy_c32'/'last_step8000.pt',map_location='cpu',weights_only=False)
        self.context_model.load_state_dict(checkpoint['model'])
        if number>=219:
            if number in [220,222]:
                from context_wide220 import ContextWide220
                self.next_model=ContextWide220().cuda().eval().requires_grad_(False)
            else:self.next_model=ContextRefine213().cuda().eval().requires_grad_(False)
            self.next_model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])
        self.small_model=CaptureFeatures(self.small_model)
        self.source='Frozen three-frame SR; causal target/surround refinement; current large-body contrast neighborhood protected'

    def reset(self):
        super().reset()
        self.tracks=CaptureStep(self.tracks);self.persistent=CaptureStep(self.persistent)
        self.surround_observations=SurroundStable211();self.previous_context_base=None;self.previous_context_raw=None
        self.context_raw_history=deque(maxlen=3)

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        self.small_model.records=[]
        before=super().native(fields,row,frame,initial=initial)
        self.last_base206_native=before.clone()
        self.last_nn_context=None
        self.last_before_native=before.clone()
        complete=before[0,0].float().cpu().numpy();body=complete[TOP:BOTTOM]
        physical=self.parent.last_physical_fore95.astype(np.float32)
        self.context_raw_history.append(physical.copy())
        self.last_context_head=np.zeros_like(body)
        if self.previous_context_base is None:
            self.previous_context_base=body.copy();self.previous_context_raw=physical.copy()
        if self.small_model.records:
            old=np.zeros((13,BOTTOM-TOP,1280),np.float32)
            for k,y in enumerate(range(0,BOTTOM-TOP,128)):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-8)
                old[:,y:end]=self.small_model.records[k][:,y-top:end-top]
            protect,_=protection167(self.tracks,self.persistent.states,physical,self.parent.gain)
            unprotected=self.persistent.last[1]
            weight=grow163(unprotected*(1-protect),protect,self.road)
            small=route180(self.persistent.states,self.persistent.previous_histories,protect,self.road)
            large=np.clip((weight-small)/np.maximum(weight,1e-6),0,1)
            owner=owner167(self.persistent.states,physical)*(1-protect)
            active=owner*large*self.road;active[:,:64]=0
            current=self.tracks.last[1]
            confirmed=np.maximum(np.maximum(owner,current),protect)
            independent=self.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            stable,_=self.surround_observations.step(body,independent,physical,confirmed,weight,self.road)
            hp=complete-cv2.blur(complete,(9,9),borderType=cv2.BORDER_REFLECT)
            support=np.maximum(old[11],active)
            feature=assemble213(old,body,hp[TOP:BOTTOM],body-self.previous_context_base,
                                physical-self.previous_context_raw,stable,body-independent,support)
            compact=np.zeros_like(weight)
            for state in self.persistent.states:
                y,x,h,w=state['box'];amount=float(np.clip((44-max(h,w))/8,0,1))
                if amount:compact=np.maximum(compact,amount*owner167([state],physical))
            compact*=self.road*(1-protect);compact[:,:64]=0
            self.last_support_mask=np.maximum(weight,compact).astype(np.float16).astype(np.float32)
            self.last_active=active.copy()
            self.last_protect=protect.copy()
            mask=np.maximum(weight,compact)*(active<=0)
            # Match the half-precision observation cache used in training.
            mask=mask.astype(np.float16).astype(np.float32)
            head=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-8);bottom=min(BOTTOM-TOP,end+8)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                masks=torch.from_numpy(mask[top:bottom])[None,None].to(before.device)
                head[y:end]=self.context_model(values,masks)[0,0,y-top:end-top].cpu().numpy()
            distance=cv2.distanceTransform((active<=0).astype(np.uint8),cv2.DIST_C,3)
            guard=np.clip((distance-6)/6,0,1)
            head*=guard
            self.last_context_head=head
            if self.number>=219:
                before216=before.clone()
                before216[...,TOP:BOTTOM,:]+=self.before_alpha*torch.from_numpy(head)[None,None].to(before.device)
                before216=before216.clamp(0,255)
                self.last_before_native=before216.clone()
                next_mask=mask*guard
                next_feature=feature.copy()
                if self.number==222:
                    released=reclaim221(self.persistent.states,list(self.context_raw_history),protect,self.road).astype(np.float16).astype(np.float32)
                    next_mask=np.maximum(next_mask,released*guard)
                    next_feature[17]=np.maximum(next_feature[17],released)
                    self.last_nn_context=next_feature.copy()
                    self.last_compact_mask=compact.copy()
                    self.last_release=released.copy()
                    self.last_guard=guard.copy()
                next_head=np.zeros_like(body)
                for y in range(0,BOTTOM-TOP,128):
                    end=min(y+128,BOTTOM-TOP);top=max(0,y-32);bottom=min(BOTTOM-TOP,end+32)
                    values=torch.from_numpy(next_feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                    masks=torch.from_numpy(next_mask[top:bottom])[None,None].to(before.device)
                    next_head[y:end]=self.next_model(values,masks)[0,0,y-top:end-top].cpu().numpy()
                proposed=before.clone()
                proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(next_head)[None,None].to(before.device)
                proposed=proposed.clamp(0,255)
                result=before216+self.context_alpha*(proposed-before216)
                self.last_context_head=next_head
            else:
                result=before.clone()
                result[...,TOP:BOTTOM,:]+=self.context_alpha*torch.from_numpy(head)[None,None].to(before.device)
        else:result=before
        self.previous_context_base=body.copy();self.previous_context_raw=physical.copy()
        return result.clamp(0,255)

    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        independent=self.parent.full(full,self.independent_native116,current,shifted,fields)
        common=self.static.full(independent)
        self.last_before_full=lift_native(common,self.last_before_native)
        return lift_native(common,native)
