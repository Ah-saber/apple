"""Protect weak positive bodies and compact cold targets, release cold body parts."""
import cv2,numpy as np
from mixed_body111 import MixedMotion111
from matched_positive104 import MatchedPositive104

class BodyTracks132:
    def __init__(self,road):
        self.road=road;self.medium=MixedMotion111(road);self.weak=MatchedPositive104(road)
    def step(self,presence,instant,noise,gate,physical,gain):
        matched=self.medium.step(presence,instant,noise)
        current=(matched>0).astype(np.uint8);past=np.zeros_like(current)
        for r in self.medium.last_matches:
            y,x,h,w=r['y'],r['x'],r['h'],r['w'];component=current[y:y+h,x:x+w]
            for lag,dy,dx in r['history']:past[y+dy:y+dy+h,x+dx:x+dx+w]|=component
        self.weak.step(presence,gate,np.zeros_like(presence),physical,gain)
        protect=np.zeros_like(current)
        extra=0
        for y,x,h,w,polarity,component,history in self.weak.last_matches:
            selected=component>0;strength=float(np.mean(np.abs(gain*physical[y:y+h,x:x+w][selected])))
            # Hot compact targets remain protected. Larger cold components can be vehicle roofs.
            if (polarity>0 and strength<=16 and (selected.sum()<=32 or max(h,w)<=12)) or (polarity<0 and selected.sum()<=9):
                protect[y:y+h,x:x+w]|=selected.astype(np.uint8)
            elif (polarity<0 and selected.sum()>9) or (polarity>0 and selected.sum()>32 and max(h,w)>12):
                current[y:y+h,x:x+w]|=selected.astype(np.uint8)
                for lag,dy,dx in history:past[y+dy:y+dy+h,x+dx:x+dx+w]|=selected.astype(np.uint8)
                extra+=1
        protect=cv2.dilate(protect,np.ones((13,13),np.uint8))
        self.last_protection=protect.copy()
        current=cv2.dilate(current,np.ones((7,7),np.uint8))*(1-protect)
        past=cv2.dilate(past,np.ones((7,7),np.uint8))*(1-protect)
        self.last_count=len(self.medium.last_matches)+extra
        return np.maximum(current,past).astype(np.float32)*self.road,current.astype(np.float32),past.astype(np.float32)

