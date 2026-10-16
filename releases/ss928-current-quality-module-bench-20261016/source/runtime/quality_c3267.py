"""Run the C32 aligned-history and current-target correction causally."""
import json
from collections import deque
import numpy as np
import cv2
import torch
from target_residual import ROOT
from quality_control43 import QualityControl43
from road_residual53 import RoadResidual53,support53
from residual50 import RawSignal50

class QualityC3267:
    def __init__(self,number=65):
        self.scene='weather_heavy_c32';self.number=number
        self.parent=QualityControl43(self.scene)
        self.raw=RawSignal50(self.scene)
        self.contrast=RoadResidual53().cuda().eval().requires_grad_(False)
        ck=torch.load(ROOT/'MOTION53-20261014'/self.scene/'last_step500.pt',map_location='cpu',weights_only=False)
        self.contrast.load_state_dict(ck['model'])
        self.alphas={n:json.loads((ROOT/('MOTION%d-20261014'%n)/self.scene/'fit.json').read_text())['best']['alpha'] for n in [55,62,63,number]}
        self.top,self.bottom=180,650;h,w=470,1280
        self.yy,self.xx=np.mgrid[:h,:w].astype(np.float32)
        self.support=support53((180,0,h,w),'cpu')[0,0].numpy();self.mask=self.support>.9
        self.reset()
        self.source='QUALITY43 + aligned55/62/63 + confirmed current contrast53 / candidate%d'%number

    def reset(self):
        self.bank=deque(maxlen=8);self.current_bank=deque(maxlen=8)
        self.previous=None;self.previous_source=None;self.first_current=None

    @staticmethod
    def local(a,size=7):
        return a-cv2.blur(a,(size,size),borderType=cv2.BORDER_REFLECT)

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset()
        v={k:a.float().cuda() for k,a in fields.items() if k!='GT'}
        source=self.parent.native(v,initial=initial)
        previous=v if self.previous is None else self.previous
        _,_,_,instant=self.parent.model(v,previous,v['baseline'],initial=True)
        top,bottom=self.top,self.bottom
        current=instant[0,0,top:bottom].clamp(0,255).cpu().numpy()
        v['raw50']=self.raw.for_record(row,frame,(0,0,1024,1280))
        learned=self.contrast(v,source,source if self.previous_source is None else self.previous_source)[0,0,top:bottom].cpu().numpy()
        path=row['raw']['path']
        physical=np.array(self.raw.open[path][frame,top:bottom],dtype=np.float32)*(255/self.raw.scale[path])
        feature=lambda a:cv2.GaussianBlur(a,(0,0),.65)-cv2.GaussianBlur(a,(0,0),7)
        a=feature(physical);detail=self.local(current)
        numerators={b:current.copy() for b in [8.,2.]};denominators={b:np.ones_like(current) for b in numerators}
        for lag in [1,2,4,8]:
            if lag>len(self.bank):continue
            past,b,old=self.bank[-lag]
            difference=self.local(physical-past,21)
            noise=max(.2,float(1.4826*np.median(np.abs(difference[self.mask]-np.median(difference[self.mask])))))
            backward=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.1,0)
            forward=cv2.calcOpticalFlowFarneback(b,a,None,.5,3,15,3,5,1.1,0)
            mx,my=self.xx+backward[...,0],self.yy+backward[...,1];magnitude=np.linalg.norm(backward,axis=2)
            inverse=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            reliable=(np.linalg.norm(backward+inverse,axis=2)<.75+.05*magnitude)&(mx>=0)&(mx<1279)&(my>=0)&(my<469)
            moving=(magnitude>.35)&(np.abs(cv2.GaussianBlur(difference,(0,0),.65))>.5*noise)&self.mask
            moving=cv2.dilate(moving.astype(np.uint8),np.ones((3,3),np.uint8))>0
            warped_raw=cv2.remap(past,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            warped_image=cv2.remap(old,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            history=np.where(moving&reliable,warped_image,old)
            residual=np.where(moving&reliable,self.local(physical-warped_raw,21),difference)
            sensor_weight=1/(1+(residual/(2*noise))**4)
            for bandwidth in numerators:
                weight=(sensor_weight/(1+((detail-self.local(history))/bandwidth)**4)/(1+lag/8)).astype(np.float32)
                weight[moving&~reliable]=0
                numerators[bandwidth]+=weight*history;denominators[bandwidth]+=weight
        source_array=source[0,0].cpu().numpy();source_detail=self.local(source_array[top:bottom])
        deltas={b:np.clip(self.local(numerators[b]/denominators[b])-source_detail,-8,8)*self.support for b in numerators}
        aligned=.5*(self.alphas[55]*deltas[8.]+self.alphas[62]*deltas[2.])*self.alphas[63]
        observation=v['current_only'][0,0,top:bottom].cpu().numpy()
        if self.first_current is None:self.first_current=observation.copy()
        history=[self.first_current]*(8-len(self.current_bank))+list(self.current_bank)
        background=np.median(np.stack(history),axis=0)
        foreground=self.local(cv2.GaussianBlur(observation-background,(0,0),.65),21)
        noise=max(.2,float(1.4826*np.median(np.abs(foreground[self.mask]-np.median(foreground[self.mask])))))
        proposal=np.clip(self.local(learned,21),-8,8)
        gate=np.clip((np.abs(foreground)-.5*noise)/noise,0,1)*(foreground*proposal>0)*self.support
        correction=aligned+gate*(proposal-aligned if self.number==66 else proposal)
        result=source_array.copy();result[top:bottom]+=self.alphas[self.number]*correction
        self.bank.append((physical.copy(),a.copy(),current.copy()));self.current_bank.append(observation.copy())
        self.previous=v;self.previous_source=source
        return torch.from_numpy(result.clip(0,255))[None,None].cuda()
