"""Fuse RAW-confirmed background; let foreground follow only its local background offset."""
from collections import deque
import cv2,numpy as np

class MotionBackground226:
    def __init__(self,blur_sigma=2.,history=4,maximum_change=3.):
        self.bank=deque(maxlen=history);self.blur_sigma=blur_sigma;self.maximum_change=maximum_change
    def step(self,image,independent,physical,confirmed,support,road,owner,weak_protect):
        core=road>.9;core[:,:64]=False;values=physical[core]
        sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
        foreground=((confirmed>0)&(np.abs(physical)>2*sigma)).astype(np.uint8)
        foreground=cv2.dilate(foreground,np.ones((3,3),np.uint8))
        tiny=(weak_protect>0)&(owner<=0)
        blocked=(foreground>0)|tiny
        valid=(~blocked).astype(np.float32)
        denominator=cv2.GaussianBlur(valid,(0,0),self.blur_sigma,borderType=cv2.BORDER_REFLECT)
        observed=cv2.GaussianBlur(image*valid,(0,0),self.blur_sigma,borderType=cv2.BORDER_REFLECT)/np.maximum(denominator,1e-3)
        total=observed.copy();count=np.ones_like(observed);union=support.copy()
        guide=cv2.GaussianBlur(physical,(0,0),.8,borderType=cv2.BORDER_REFLECT)
        for lag,(old,old_guide,old_blocked,old_support,old_independent) in enumerate(reversed(self.bank),1):
            raw_weight=np.clip(1-np.abs(guide-old_guide)/(2*sigma),0,1)
            difference=independent-old_independent;shift=float(np.median(difference[core]))
            sr_sigma=max(.5,float(1.4826*np.median(np.abs(difference[core]-shift))))
            sr_weight=np.clip(1-np.abs(difference-shift)/(2*sr_sigma),0,1)
            weight=raw_weight*sr_weight*(~old_blocked)*old_support*(.75**lag)
            total+=weight*old;count+=weight;union=np.maximum(union,weight)
        delta=np.clip(total/count-observed,-self.maximum_change,self.maximum_change)*np.clip(union,0,1)*road
        delta[blocked]=0
        shifted=np.zeros_like(image);coverage=np.zeros_like(image);components=0
        all_strong=(owner>0)&(np.abs(physical)>2*sigma)
        for polarity in [1,-1]:
            seeds=(all_strong&(polarity*physical>0)).astype(np.uint8)
            n,labels,boxes,_=cv2.connectedComponentsWithStats(seeds,connectivity=8)
            for label in range(1,n):
                x,y,w,h,area=boxes[label]
                if area<4:continue
                top,left=max(0,y-6),max(0,x-6);bottom,right=min(len(labels),y+h+6),min(labels.shape[1],x+w+6)
                crop=np.s_[top:bottom,left:right];component=(labels[crop]==label).astype(np.uint8)
                outer=cv2.dilate(component,np.ones((13,13),np.uint8));inner=cv2.dilate(component,np.ones((5,5),np.uint8))
                ring=(outer>0)&(inner==0)&(~blocked[crop])&(road[crop]>.9)
                if ring.sum()<12:continue
                offset=float(np.median((image[crop]+delta[crop])[ring])-np.median(image[crop][ring]))
                target=(cv2.dilate(component,np.ones((3,3),np.uint8))>0)&(foreground[crop]>0)
                shifted[crop]+=offset*target;coverage[crop]+=target;components+=1
        selected=coverage>0;delta[selected]=shifted[selected]/coverage[selected]
        delta[tiny]=0;delta[:,:64]=0
        self.bank.append((observed.copy(),guide.copy(),blocked.copy(),support.copy(),independent.copy()))
        return delta,dict(history=len(self.bank),changed_pixels=int((np.abs(delta)>1e-5).sum()),
                          foreground_offset_components=components,maximum=float(np.abs(delta).max()))
