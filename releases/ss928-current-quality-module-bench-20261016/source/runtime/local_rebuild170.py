"""Anchor physical reconstruction with stable background inside each repair region."""
import cv2,numpy as np

def extra170(physical_background,sr_background,current_foreground,source,weight,road,gains,raw_physical):
    response=np.where(current_foreground>=0,gains['1'],gains['-1'])*current_foreground
    desired=physical_background+response;difference=physical_background-sr_background
    core=road>.9;core[:,:64]=False
    values=raw_physical[core]
    sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
    count,labels,_,_=cv2.connectedComponentsWithStats((weight>0).astype(np.uint8),connectivity=8)
    extra=np.zeros_like(source,dtype=np.float32);records=[]
    for label in range(1,count):
        component=(labels==label).astype(np.uint8)
        stable=(component>0)&core&(np.abs(raw_physical)<sigma)
        pixels=int(stable.sum());anchor='interior_stable'
        if pixels<12:
            outer=cv2.dilate(component,np.ones((31,31),np.uint8));inner=cv2.dilate(component,np.ones((11,11),np.uint8))
            stable=(outer>inner)&(weight==0)&core;pixels=int(stable.sum());anchor='outer_fallback'
        if pixels<12:
            records.append(dict(component=label,anchor_pixels=pixels,status='retain161'));continue
        offset=float(np.median(difference[stable]));selected=component>0
        extra[selected]=weight[selected]*np.clip(desired[selected]-offset-source[selected],-24,24)
        records.append(dict(component=label,anchor=anchor,anchor_pixels=pixels,offset=offset,sigma=sigma))
    return extra,records
