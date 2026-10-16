"""Confirm a current body from coherent physical RAW, including weak SR bodies."""
import cv2
import numpy as np

def physical_body_gate80(presence,current,mask):
    p=cv2.GaussianBlur(presence.astype(np.float32),(0,0),.65)
    now=cv2.GaussianBlur(current.astype(np.float32),(0,0),.65)
    def sigma(a):return max(.2,float(1.4826*np.median(np.abs(a[mask]-np.median(a[mask])))))
    pn,nn=sigma(p),sigma(now)
    agree=p*now>0
    seeds=agree&(np.abs(p)>pn)&(np.abs(now)>.5*nn)
    n,labels,_,_=cv2.connectedComponentsWithStats(seeds.astype(np.uint8),connectivity=8)
    strong=(np.abs(p)>1.5*pn)&(np.abs(now)>1.5*nn)&agree
    counts=np.bincount(labels[strong],minlength=n)
    keep=(counts[labels]>=2)&(labels!=0)
    keep=cv2.dilate(keep.astype(np.uint8),np.ones((3,3),np.uint8))
    gate=np.clip((np.abs(p)-.5*pn)/pn,0,1)*np.clip((np.abs(now)-.5*nn)/nn,0,1)
    return (gate*agree*keep*np.sign(p)).astype(np.float32)
