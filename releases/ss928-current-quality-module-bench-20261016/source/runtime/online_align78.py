"""Causal RAW foreground alignment used by the original-input controller."""
from collections import deque
import cv2
import numpy as np
from road_residual53 import support53

class ForegroundFusion78:
    def __init__(self):
        self.support=support53((176,0,478,1280),'cpu')[0,0].numpy()
        self.yy,self.xx=np.mgrid[:478,:1280].astype(np.float32)
        self.reset()
    def reset(self):self.bank=deque(maxlen=4)
    def step(self,raw,body,raw_noise,fore,presence,coherent,sr_noise):
        smooth=cv2.GaussianBlur(raw,(0,0),.65)
        numerator=body.copy();denominator=np.ones_like(body)
        for lag in [1,2,4]:
            if len(self.bank)<lag:continue
            old_raw,old_body,old_noise=self.bank[-lag]
            backward=cv2.calcOpticalFlowFarneback(smooth,old_raw,None,.5,3,15,3,5,1.1,0)
            forward=cv2.calcOpticalFlowFarneback(old_raw,smooth,None,.5,3,15,3,5,1.1,0)
            mx,my=self.xx+backward[...,0],self.yy+backward[...,1]
            inverse=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            valid=(np.linalg.norm(backward+inverse,axis=2)<.75+.05*np.linalg.norm(backward,axis=2))&(mx>=0)&(mx<1279)&(my>=0)&(my<477)
            previous_raw=cv2.remap(old_raw,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            previous_body=cv2.remap(old_body,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            ratio=np.clip(np.abs(smooth)/np.maximum(np.abs(previous_raw),old_noise),.25,2.)
            estimate=previous_body*ratio
            reliability=valid&(smooth*previous_raw>0)&(previous_raw*previous_body>0)
            weight=np.clip((np.abs(previous_raw)-old_noise)/old_noise,0,1)*reliability/(1+lag/4)
            numerator+=weight*estimate;denominator+=weight
        gate=((fore*presence>0)&(raw*presence>0)).astype(np.float32)
        gate*=np.clip((np.abs(fore)-.5*sr_noise)/sr_noise,0,1)
        gate*=np.clip((np.abs(raw)-.5*raw_noise)/raw_noise,0,1)*coherent
        result=np.clip(numerator/denominator-body,-6,6)*gate*self.support
        self.bank.append((smooth.copy(),body.copy(),float(raw_noise)))
        return result
