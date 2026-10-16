"""Aligned current/past RAW cues for current compact targets, without output feedback."""
from collections import deque
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from body_ownership167 import owner167
from temporal_body194 import sigma194
from small_residual249 import SmallResidual249

CACHE258=Path('/tmp/rawir-small258-20261016')

class RawSmallTemporal258:
    def __init__(self):self.history=deque(maxlen=4)
    def amount(self,h,w):return float(np.clip((44-max(h,w))/8,0,1))
    def align(self,old,xx,yy,dx,dy,current,owned,sigma):
        return cv2.remap(old,xx+dx,yy+dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    def brightness_offset(self,current,old,selected):
        return float(np.median((current-old)[selected]))
    def step(self,physical,states,road):
        self.history.append(physical.copy());sigma=sigma194(physical,road)
        result=np.zeros((4,*physical.shape),np.float32);coverage=np.zeros_like(physical)
        for state in states:
            y,x,h,w=state['box'];amount=self.amount(h,w)
            if amount<=0:continue
            top,left=max(0,y-4),max(0,x-4);bottom,right=min(len(physical),y+h+4),min(physical.shape[1],x+w+4)
            crop=np.s_[top:bottom,left:right];current=physical[crop]
            owned=owner167([state],physical)[crop]*road[crop]*amount
            total=current.copy();square=np.square(current);count=np.ones_like(current)
            yy,xx=np.mgrid[top:bottom,left:right].astype(np.float32)
            positions=list(state['history'])
            for lag in range(1,min(4,len(self.history),len(positions))):
                py,px,ph,pw=positions[-(lag+1)]
                dy=py+ph/2-y-h/2;dx=px+pw/2-x-w/2
                old=self.align(self.history[-(lag+1)],xx,yy,dx,dy,current,owned,sigma)
                matching=(current*old>0)&(np.abs(old)>sigma)
                adjusted=old.copy();valid=np.zeros_like(current,dtype=bool)
                for sign in [1,-1]:
                    group=(owned>0)&(sign*current>sigma)
                    selected=group&matching&(np.abs(current)>2*sigma)
                    if selected.sum()<4:continue
                    offset=self.brightness_offset(current,old,selected);adjusted[group]+=offset
                    valid|=group&matching
                weight=np.clip(1-np.abs(adjusted-current)/(3*sigma),0,1)*valid
                total+=weight*adjusted;square+=weight*np.square(adjusted);count+=weight
            mean=total/count;deviation=np.sqrt(np.maximum(square/count-np.square(mean),0))
            cue=np.stack([mean/64,(mean-current)/4,deviation/sigma,(count-1)/3])
            result[:,top:bottom,left:right]+=cue*owned;coverage[crop]+=owned
        result=np.divide(result,coverage[None],out=np.zeros_like(result),where=coverage[None]>0)
        result[:,:,:64]=0
        return result.clip(-4,4).astype(np.float16)

def features258(feature,raw_cues):
    return np.concatenate([np.asarray(feature,np.float32),np.asarray(raw_cues,np.float32)],axis=0).clip(-4,4).astype(np.float16)

class SmallTemporal258(SmallResidual249):
    def __init__(self):
        super().__init__();self.net[0]=nn.Conv2d(23,32,3,padding=1,padding_mode='reflect')
    def initialize_temporal(self,state):
        with torch.no_grad():
            self.net[0].weight[:,:19].copy_(state['net.0.weight']);self.net[0].weight[:,19:].zero_()
        self.load_state_dict({k:v for k,v in state.items() if k!='net.0.weight'},strict=False)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
