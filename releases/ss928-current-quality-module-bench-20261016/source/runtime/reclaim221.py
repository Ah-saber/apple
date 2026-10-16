"""Release guard spill only on owned compact bodies supported by three RAW frames."""
import cv2,numpy as np
from body_ownership167 import owner167

def reclaim221(states,history,protect,road):
    current=history[-1];core=road>.9;core[:,:64]=False
    values=current[core];sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
    released=np.zeros_like(current,dtype=np.float32)
    if len(history)<3:return released
    for state in states:
        if len(state['history'])<3:continue
        y,x,h,w=state['box'];amount=float(np.clip((44-max(h,w))/8,0,1))
        if amount<=0:continue
        top,left=max(0,y-3),max(0,x-3);bottom,right=min(len(current),y+h+3),min(current.shape[1],x+w+3)
        crop=np.s_[top:bottom,left:right];value=current[crop];polarity=np.sign(value)
        supported=np.abs(value)>2*sigma
        yy,xx=np.mgrid[top:bottom,left:right].astype(np.float32)
        for lag in [1,2]:
            py,px,ph,pw=state['history'][-(lag+1)]
            dy=round(py+ph/2-y-h/2);dx=round(px+pw/2-x-w/2)
            old=cv2.remap(history[-(lag+1)],xx+dx,yy+dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            supported &= polarity*old>sigma
        owner=owner167([state],current)[crop]
        released[crop]=np.maximum(released[crop],amount*owner*supported*(protect[crop]>0))
    released*=road;released[:,:64]=0
    return released
