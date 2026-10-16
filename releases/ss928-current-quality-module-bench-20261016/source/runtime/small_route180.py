"""Restore the proven neural treatment for smaller RAW bodies and their recent footprints."""
import cv2,numpy as np
from halo_weight163 import grow163

def route180(states,histories,protect,road):
    mask=np.zeros_like(road);height,width=mask.shape
    for state in states:
        y,x,h,w=state['box'];amount=float(np.clip((36-min(h,w))/8,0,1))
        if amount==0:continue
        positions=[(y,x)];history=histories.get(id(state),[])
        for lag in [1,2,4]:
            if len(history)<lag:continue
            py,px,ph,pw=history[-lag]
            oy=int(round(py+ph/2-h/2));ox=int(round(px+pw/2-w/2))
            if oy<0 or ox<0 or oy+h>height or ox+w>width:continue
            positions.append((oy,ox))
        for py,px in positions:mask[py:py+h,px:px+w]=np.maximum(mask[py:py+h,px:px+w],amount*state['component'])
    mask=cv2.dilate(mask,np.ones((7,7),np.uint8))*(1-protect)*road;mask[:,:64]=0
    return grow163(mask,protect,road)
