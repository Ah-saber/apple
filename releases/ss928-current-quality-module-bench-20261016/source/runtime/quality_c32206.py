"""Deliver trained205 current-small repair above independent complete198."""
import json
from collections import deque
import cv2,numpy as np,torch
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_c32108 import QualityC32108
from quality_c32112 import StaticLowpass112
from body_current116 import CurrentRepair116,features116
from body_tracks132 import BodyTracks132
from persistent_body154 import PersistentBodies154
from body_ownership167 import protection167,owner167
from halo_weight163 import grow163
from local_rebuild170 import extra170
from physical_reference90 import detail,clean_common
from small_route180 import route180
from background_time191 import filter191
from temporal_body194 import filter194
from refine176 import features176
from gate205 import Gate205,features205
from quality_control43 import lift_native

class QualityC32206:
    def __init__(self,number=206):
        if number!=206:raise ValueError('This controller is round206')
        self.parent=QualityC32108(108);self.road=self.parent.road
        fit161=json.loads((ROOT/'MOTION161-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        fit170=json.loads((ROOT/'MOTION170-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        if fit161['status']!='passed' or fit170['status']!='passed':raise ValueError('Original training selections required')
        self.alpha161=fit161['head_alpha161'];self.alpha170=fit170['head_alpha170']
        fit195=json.loads((ROOT/'MOTION195-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        fit197=json.loads((ROOT/'MOTION197-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        response=json.loads((ROOT/'MOTION195-20261014'/'weather_heavy_c32'/'negative_response.json').read_text())['best']
        if fit195['status']!='passed' or fit197['status']!='passed' or not response['enabled']:raise ValueError('Combination197 training selection required')
        self.alpha195=fit195['head_alpha195'];self.alpha197=fit197['head_alpha197'];self.negative_increment=response['negative_increment']
        fit132=json.loads((ROOT/'MOTION132-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        fit180=json.loads((ROOT/'MOTION180-20261014'/'weather_heavy_c32'/'head_selection.json').read_text())
        self.alpha132=fit132['head_alpha132'];self.beta=fit180['diagnostic_best']['alpha']
        self.gains=json.loads((ROOT/'MOTION156-20261014'/'weather_heavy_c32'/'response.json').read_text())['gains']
        self.model=CurrentRepair116().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/'MOTION132-20261014'/'weather_heavy_c32'/'last_step1500.pt',map_location='cpu',weights_only=False)['model'])
        fit205=json.loads((ROOT/'MOTION205-20261014'/'weather_heavy_c32'/'fit.json').read_text())
        if fit205['status']!='passed':raise ValueError('Training205 selection required')
        self.alpha205=fit205['head_alpha205']
        self.small_model=Gate205().cuda().eval().requires_grad_(False)
        self.small_model.load_state_dict(torch.load(ROOT/'MOTION205-20261014'/'weather_heavy_c32'/'last_step2000.pt',map_location='cpu',weights_only=False)['model'])
        self.center=self.road>.9;self.center[:,:64]=False
        self.source='Original SR3 -> complete198 independent body and boundary -> bounded trained205 current-small gate'
        self.reset()
    def reset(self):
        self.bank=deque(maxlen=64);self.time_bank=deque(maxlen=4);self.body_time_bank=deque(maxlen=2);self.persistent=PersistentBodies154(self.road);self.tracks=BodyTracks132(self.road);self.static=StaticLowpass112();self.start_frame=None
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset()
        gate_context=None;proposal205=None
        if self.start_frame is None:self.start_frame=frame
        self.parent.native(fields,row,frame,initial=initial)
        image=self.parent.last_body_image107.copy();body=image[TOP:BOTTOM].copy()
        raw_controller=self.parent.parent.parent.parent
        values=raw_controller.last_body_fields
        presence=values['presence76'][0,0].float().cpu().numpy()
        instant=values['instant_presence76'][0,0].float().cpu().numpy()
        noise=float(values['noise76'][0].float().cpu());gate=self.parent.last_gate104;physical=self.parent.last_physical_fore95
        mask,current,past=self.tracks.step(presence,instant,noise,gate,physical,self.parent.gain)
        mask=mask.astype(np.float16).astype(np.float32)
        if len(self.bank)>=16:
            background=np.median(np.stack(self.bank),axis=0);foreground=body-background;foreground-=np.median(foreground[self.center])
            corrected=fields['current_only'][0,0,TOP:BOTTOM].float().cpu().numpy()
            plain=fields['plain_current116'][0,0,TOP:BOTTOM].float().cpu().numpy()
            features=features116(body,corrected,plain,physical,foreground,presence,instant,current,past,self.road,self.parent.gain)
            head=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-8);bottom=min(BOTTOM-TOP,end+8)
                values=torch.from_numpy(features[:,top:bottom].astype(np.float32))[None].cuda()
                gate_tensor=torch.from_numpy(mask[top:bottom].astype(np.float32))[None,None].cuda()
                head[y:end]=self.model(values,gate_tensor)[0,0,y-top:end-top].cpu().numpy()
            full_head=np.zeros_like(image);full_head[TOP:BOTTOM]=head
            contrast=full_head-cv2.blur(full_head,(21,21),borderType=cv2.BORDER_REFLECT)
            old_protect=self.tracks.last_protection
            histories={id(s):list(s['history']) for s in self.persistent.states}
            unmasked,unprotected,_,_=self.persistent.step(physical.astype(np.float32),self.tracks,contrast[TOP:BOTTOM],self.parent.gain,np.zeros_like(old_protect))
            tracked=(unmasked*(1-old_protect)).astype(np.float16).astype(np.float32)
            old_weight=(unprotected*(1-old_protect)).astype(np.float16).astype(np.float32)
            response=np.where(physical>=0,self.gains['1'],self.gains['-1'])*physical
            replacement=np.maximum(mask,tracked)*np.clip(response-16*features[7].astype(np.float32),-24,24)
            fixed_head=(1-old_weight)*head+old_weight*.5*replacement
            image[TOP:BOTTOM]+=self.alpha161*fixed_head
            image=image.clip(0,255)
            fixed_body=image[TOP:BOTTOM].copy()
            protect,_=protection167(self.tracks,self.persistent.states,physical,self.parent.gain)
            weight=grow163(unprotected*(1-protect),protect,self.road)
            filtered=cv2.GaussianBlur(physical,(0,0),.8,borderType=cv2.BORDER_REFLECT)
            raw=raw_controller.raw.open[row['raw']['path']]
            raw_background=np.median(np.asarray(raw[max(self.start_frame,frame-64):frame,TOP:BOTTOM],np.float32),axis=0)
            physical_background=self.parent.tone.apply(detail(clean_common(raw_background,self.parent.reference)))
            extra,_=extra170(physical_background,background,filtered,image[TOP:BOTTOM],weight,self.road,self.gains,physical)
            image[TOP:BOTTOM]+=self.alpha170*extra
            base=image[TOP:BOTTOM].clip(0,255)
            small=route180(self.persistent.states,histories,protect,self.road)
            large=np.clip((weight-small)/np.maximum(weight,1e-6),0,1)
            owner=owner167(self.persistent.states,physical)*(1-protect)
            active=owner*large*self.road;active[:,:64]=0
            bilateral,_=filter194(physical.astype(np.float32),self.body_time_bank,active,self.road)
            self.body_time_bank.append(physical.astype(np.float32).copy())
            body_extra,_=extra170(physical_background,background,bilateral,fixed_body,weight,self.road,self.gains,physical)
            unweighted=np.divide(body_extra,weight,out=np.zeros_like(weight),where=weight>0)
            reference=np.clip(fixed_body+unweighted,0,255)
            reference=np.clip(reference+self.negative_increment*np.minimum(bilateral,0),0,255)
            current_delta=active*np.clip(reference-base,-8,8)
            confirmed=np.maximum(owner,current)
            denoised,_=filter191(filtered,self.time_bank,confirmed,self.road)
            background_extra,_=extra170(physical_background,background,denoised,fixed_body,weight,self.road,self.gains,physical)
            background_delta=np.clip(fixed_body+self.alpha170*background_extra,0,255)-base
            background_delta[(weight<=0)|(confirmed>0)]=0;background_delta[:,:64]=0
            self.time_bank.append(filtered.copy())
            small_delta=small*(np.clip(body+self.alpha132*head,0,255)-base)
            image[TOP:BOTTOM]=base+self.alpha197*(self.alpha195*current_delta+self.beta*small_delta+(1-self.beta*small)*background_delta)
            current_small=small*owner;current_small[:,:64]=0
            proposal205=(1-self.beta)*current_small*(np.clip(body+self.alpha132*head,0,255)-base)
            gate_context=features176(base,fixed_body,physical_background,background,physical,extra,weight,features[10].astype(np.float32),features[11].astype(np.float32),self.gains)
        self.bank.append(body)
        image=image.clip(0,255);image[:,:3]+=self.parent.last_edge_extra106
        self.independent_native116=torch.from_numpy(image.clip(0,255))[None,None].to(fields['baseline'].device)
        self.last_source92=self.parent.last_source92
        base198=self.static.native(self.independent_native116,fields,gate,self.road).clamp(0,255)
        result=base198.clone()
        if gate_context is not None:
            complete=base198[0,0].float().cpu().numpy()
            hp=complete-cv2.blur(complete,(9,9),borderType=cv2.BORDER_REFLECT)
            gate_context[0]=complete[TOP:BOTTOM]/255
            gate_context[9]=hp[TOP:BOTTOM]/16
            gate_context=features205(gate_context,proposal205)
            learned=np.zeros_like(proposal205)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-8);bottom=min(BOTTOM-TOP,end+8)
                values=torch.from_numpy(gate_context[:,top:bottom])[None].cuda()
                proposed=torch.from_numpy(proposal205[top:bottom].astype(np.float32))[None,None].cuda()
                learned[y:end]=self.small_model(values,proposed)[0,0,y-top:end-top].cpu().numpy()
            result[...,TOP:BOTTOM,:]+=self.alpha205*torch.from_numpy(learned)[None,None].to(result.device)
        return result.clamp(0,255)
    @torch.inference_mode()
    def full(self,full,native,current,shifted,fields):
        base198=self.static.full(self.parent.full(full,self.independent_native116,current,shifted,fields))
        return lift_native(base198,native)

