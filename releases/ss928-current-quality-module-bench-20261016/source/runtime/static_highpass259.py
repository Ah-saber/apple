"""Causal static-detail stabilization; current RAW target neighborhoods stay protected."""
from collections import deque
import cv2,numpy as np
from compact_match93 import TOP,BOTTOM

def motion259(activity,occupancy,owner,current,protect,binary=False,raw_gate=None):
    motion=np.maximum(activity,occupancy).astype(np.float32)
    if binary:motion=(motion>.5).astype(np.float32)
    foreground=(np.maximum(np.maximum(owner,current),protect)>0).astype(np.uint8)
    foreground=cv2.dilate(foreground,np.ones((49,49),np.uint8))
    if raw_gate is not None:
        raw_confirmed=cv2.dilate((np.abs(raw_gate)>0).astype(np.uint8),np.ones((25,25),np.uint8))
        foreground=np.maximum(foreground,raw_confirmed)
    motion[TOP:BOTTOM]=np.maximum(motion[TOP:BOTTOM],foreground)
    motion[:,:3 if raw_gate is not None else 64]=1
    return motion.clip(0,1)

class StaticHighpass259:
    def __init__(self,preserve_peaks=False,preserve_compact=False,protected_columns=64):
        self.history=deque(maxlen=4);self.preserve_peaks=preserve_peaks;self.preserve_compact=preserve_compact;self.protected_columns=protected_columns
    def step(self,image,motion):
        hp=image-cv2.blur(image,(9,9),borderType=cv2.BORDER_REFLECT)
        union=motion.copy()
        for _,old_motion in self.history:union=np.maximum(union,old_motion)
        allowed=1-union;allowed[:,:self.protected_columns]=0
        if self.preserve_peaks:
            broad=image-cv2.blur(image,(21,21),borderType=cv2.BORDER_REFLECT)
            peak=np.clip((np.maximum(np.abs(hp),np.abs(broad))-1)/3,0,1)
            allowed*=1-peak
        total=hp.copy();count=np.ones_like(hp)
        sample=allowed[::4,::4]>.8
        for old,_ in self.history:
            difference=hp-old;values=difference[::4,::4][sample]
            sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values))))) if values.size else .2
            weight=np.exp(-.5*np.square(difference/(3*sigma))).astype(np.float32)
            total+=weight*old;count+=weight
        delta=np.clip(total/count-hp,-3,3)*allowed
        if self.preserve_compact:
            broad=image-cv2.blur(image,(21,21),borderType=cv2.BORDER_REFLECT)
            seeds=[((sign*hp>2.5)|(sign*broad>3.5)).astype(np.uint8) for sign in [1,-1]]
            exclusion=(seeds[0]|seeds[1])>0
            correction=np.zeros_like(image);coverage=np.zeros_like(image)
            for seed in seeds:
                n,labels,boxes,_=cv2.connectedComponentsWithStats(seed,8)
                for label in range(1,n):
                    x,y,w,h,area=boxes[label]
                    if not(2<=area<=128 and w<=24 and h<=24):continue
                    top,left=max(0,y-7),max(0,x-7);bottom,right=min(len(image),y+h+7),min(image.shape[1],x+w+7)
                    crop=np.s_[top:bottom,left:right];core=(labels[crop]==label).astype(np.uint8)
                    target=cv2.dilate(core,np.ones((3,3),np.uint8))>0
                    ring=(cv2.dilate(core,np.ones((13,13),np.uint8))>0)&(cv2.dilate(core,np.ones((5,5),np.uint8))==0)&(~exclusion[crop])&(allowed[crop]>.5)
                    if ring.sum()<12 or float(allowed[crop][target].mean())<.5:continue
                    old=image[crop];new=old+delta[crop]
                    old_gap=float(old[target].mean()-np.median(old[ring]))
                    offset=float(np.median(new[ring])+old_gap-new[target].mean())
                    weight=target*allowed[crop];weight/=max(float(weight[target].mean()),1e-6)
                    correction[crop]+=offset*weight;coverage[crop]+=target
            delta+=np.divide(correction,coverage,out=np.zeros_like(correction),where=coverage>0)
            delta=np.clip(delta,-3,3)
        self.history.append((hp.copy(),motion.copy()))
        return delta
