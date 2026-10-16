"""Run candidate75 and signed body77 on the original SR stream."""
import json
from collections import deque
import numpy as np
import cv2
import torch
from target_residual import ROOT
from quality_c3267 import QualityC3267
from history_residual71 import HistoryResidual71,history_fields71
from body_gain76 import history_fields76
from body_gain77 import BodyGain77
from body_gain79 import BodyGain79
from body_gain80 import BodyGain80
from online_align78 import ForegroundFusion78
from gate_body80 import physical_body_gate80
from road_residual53 import support53

class QualityC3278:
    def __init__(self,number=75,export_features=False):
        self.export_features=export_features
        self.number=number;self.scene='weather_heavy_c32';self.parent=QualityC3267(65)
        self.raw=self.parent.raw;self.top,self.bottom=176,654;self.crop=(176,0,478,1280)
        self.history_model=HistoryResidual71().cuda().eval().requires_grad_(False)
        self.history_model.load_state_dict(torch.load(ROOT/'MOTION71-20261014'/self.scene/'last_step500.pt',map_location='cpu',weights_only=False)['model'])
        self.alpha75=json.loads((ROOT/'MOTION75-20261014'/self.scene/'fit.json').read_text())['best']['alpha']
        self.body_model=None;self.alpha_body=0.
        if number in [77,79,80]:
            self.body_model={77:BodyGain77,79:BodyGain79,80:BodyGain80}[number]().cuda().eval().requires_grad_(False)
            self.body_model.load_state_dict(torch.load(ROOT/('MOTION%d-20261014'%number)/self.scene/'last_step500.pt',map_location='cpu',weights_only=False)['model'])
            fit=json.loads((ROOT/('MOTION%d-20261014'%number)/self.scene/'fit.json').read_text())
            self.alpha_body=(fit['best'] if fit['status']=='passed' else fit['diagnostic_best'])['alpha']
        self.mask=support53(self.crop,'cpu')[0,0].numpy()>.9
        self.mask75=support53((180,0,470,1280),'cpu')[0,0].numpy()>.9
        self.aligned=ForegroundFusion78()
        self.source='Original SR3 -> quality65 -> RAW-confirmed history75'+(' -> body%d'%number if self.body_model is not None else '')
        self.reset()
    def reset(self):
        self.bank=deque(maxlen=64);self.previous65=None;self.previous75=None
        if hasattr(self,'aligned'):self.aligned.reset()
    @staticmethod
    def hp(a,size=21):return a-cv2.blur(a,(size,size),borderType=cv2.BORDER_REFLECT)
    def sigma(self,a):return max(.2,float(1.4826*np.median(np.abs(a[self.mask]-np.median(a[self.mask])))))
    def reflected_fields(self,fields):
        v={k:a[...,self.top:self.bottom,:].float().cuda().clone() for k,a in fields.items() if k!='GT'}
        for key in ['baseline','current_only','gated_history','past_background']:v[key][...,:3]=v[key][...,3:6].flip(-1)
        v['crop53']=self.crop
        return v
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset()
        source65=self.parent.native(fields,row,frame,initial=initial)
        observation=fields['current_only'][0,0,self.top:self.bottom].float().cpu().numpy()
        values=np.stack(list(self.bank)+[observation]);i=len(values)-1
        image65=source65[0,0,self.top:self.bottom].cpu().numpy();reference65=image65.copy();reference65[:,:3]=reference65[:,3:6][:,::-1]
        v=self.reflected_fields(fields);v['raw50']=self.raw.for_record(row,frame,self.crop)
        v['history71']=history_fields71(values,i,(0,0,478,1280),reference65,repair_boundary=True)
        s65=torch.from_numpy(reference65)[None,None].cuda()
        proposal=self.history_model(v,s65,s65 if self.previous65 is None else self.previous65)[0,0].cpu().numpy()
        physical=self.raw.open[row['raw']['path']];scale=255/self.raw.scale[row['raw']['path']]
        ids=np.maximum(np.arange(frame-10,frame-2),0);recent=np.maximum(np.arange(frame-2,frame+1),0)
        bg=np.mean(np.array(physical[ids,180:650],dtype=np.float32),axis=0)
        difference=(np.mean(np.array(physical[recent,180:650],dtype=np.float32),axis=0)-bg)*scale
        instantaneous=(physical[frame,180:650].astype(np.float32)-bg)*scale
        difference=cv2.GaussianBlur(self.hp(difference),(0,0),.65)
        instantaneous=cv2.GaussianBlur(self.hp(instantaneous),(0,0),.65)
        noise=max(.2,float(1.4826*np.median(np.abs(difference[self.mask75]-np.median(difference[self.mask75])))))
        n,labels,_,_=cv2.connectedComponentsWithStats((np.abs(difference)>noise).astype(np.uint8),connectivity=8)
        counts=np.bincount(labels[np.abs(difference)>1.5*noise],minlength=n)
        coherent=(counts[labels]>=2)&(labels!=0)
        p=proposal[4:-4];gate=np.clip((np.abs(difference)-.5*noise)/noise,0,1)*((difference*p>0)&(instantaneous*p>0))
        result=source65[0,0].cpu().numpy().copy();result[180:650]+=self.alpha75*gate*p*coherent;result=result.clip(0,255)
        if self.body_model is not None:
            reference75=result[self.top:self.bottom].copy();reference75[:,:3]=reference75[:,3:6][:,::-1]
            ids=np.arange(max(0,frame-64),frame) if frame else np.array([0])
            bg=np.median(np.array(physical[ids,self.top:self.bottom],dtype=np.float32),axis=0)
            presence=self.hp((np.mean(np.array(physical[recent,self.top:self.bottom],dtype=np.float32),axis=0)-bg)*scale)
            now=self.hp((physical[frame,self.top:self.bottom].astype(np.float32)-bg)*scale)
            raw_noise=self.sigma(presence)
            history=history_fields76(values,i,(0,0,478,1280),reference75,device='cpu',repair_boundary=True)
            sr_noise=self.sigma(history[0,1].numpy()*16)
            smooth=cv2.GaussianBlur(presence,(0,0),.65)
            changes=np.maximum(np.abs(smooth),np.abs(cv2.GaussianBlur(v['raw50'][0,12].cpu().numpy()*16,(0,0),.65)))
            smooth_noise=self.sigma(smooth)
            n,labels,_,_=cv2.connectedComponentsWithStats((changes>smooth_noise).astype(np.uint8),connectivity=8)
            counts=np.bincount(labels[changes>1.5*smooth_noise],minlength=n)
            keep=(counts[labels]>=2)&(labels!=0);keep=cv2.dilate(keep.astype(np.uint8),np.ones((17,17),np.uint8))
            presence_q=presence.astype(np.float16).astype(np.float32);now_q=now.astype(np.float16).astype(np.float32)
            for key,a in [('presence76',presence_q),('instant_presence76',now_q)]:v[key]=torch.from_numpy(a)[None,None].cuda()
            v['coherent76']=torch.from_numpy(keep.astype(np.float32))[None,None].cuda()
            v['noise76']=torch.tensor([raw_noise,sr_noise],dtype=torch.float32,device='cuda');v['history76']=history.cuda()
            if self.number in [79,80] or self.export_features:
                correction78=self.aligned.step(now_q,history[0,2].numpy()*16,raw_noise,history[0,1].numpy()*16,presence_q,keep.astype(np.float32),sr_noise)
                v['aligned78']=torch.from_numpy(correction78)[None,None].cuda()
            if self.number==80 or self.export_features:
                signed_gate=physical_body_gate80(presence_q,now_q,self.mask).astype(np.float16).astype(np.float32)
                v['signed_gate80']=torch.from_numpy(signed_gate)[None,None].cuda()
            if self.export_features:self.last_body_fields=dict(v)
            s75=torch.from_numpy(reference75)[None,None].cuda()
            correction=self.body_model(v,s75,s75 if self.previous75 is None else self.previous75)[0,0].cpu().numpy()
            result[self.top:self.bottom]+=self.alpha_body*correction;self.previous75=s75
        self.bank.append(observation.copy());self.previous65=s65
        return torch.from_numpy(result.clip(0,255))[None,None].cuda()
