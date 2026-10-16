"""Causal foreground-excluded low-frequency averaging of frozen parent outputs."""
from collections import deque
import cv2,numpy as np

class BackgroundTemporal295:
    def __init__(self,normalize=True,lowpass=9):self.history=deque(maxlen=4);self.normalize=normalize;self.lowpass=lowpass
    def reset(self):self.history.clear()
    def step(self,source,compact,active,release,road,ready=True):
        if not ready:self.reset();return np.zeros_like(source)
        foreground=(np.maximum(compact,active)>.15).astype(np.uint8)
        valid=(np.maximum(foreground,release)<.1).astype(np.float32)
        size=(self.lowpass,self.lowpass)
        count=cv2.blur(valid,size,borderType=cv2.BORDER_REFLECT)
        low=cv2.blur(source*valid,size,borderType=cv2.BORDER_REFLECT)/np.maximum(count,1e-6)
        usable=(valid>0)&(count>.2)&(road>.1)
        total=low.copy();denominator=np.ones_like(source)
        for old,old_valid in self.history:
            common=usable&old_valid
            offset=float(np.median(low[common]-old[common])) if self.normalize and int(common.sum())>=64 else 0.
            total+=np.where(old_valid,old+offset,0);denominator+=old_valid
        delta=np.where(usable&(denominator>1),np.clip(total/denominator-low,-4,4),0)*road
        # Carry the surrounding tone shift through each current foreground component.
        n,labels,_,_=cv2.connectedComponentsWithStats(foreground,8)
        for k in range(1,n):
            body=(labels==k).astype(np.uint8)
            near=cv2.dilate(body,np.ones((25,25),np.uint8))-cv2.dilate(body,np.ones((7,7),np.uint8))
            ring=(near>0)&usable
            if int(ring.sum())>=12:delta[body>0]=float(np.median(delta[ring]))*road[body>0]
        self.history.append((low.copy(),usable.copy()))
        return delta.astype(np.float32)
