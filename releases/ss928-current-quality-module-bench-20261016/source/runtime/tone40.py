"""Stabilize regional tone without averaging target pixels across frames."""
from collections import deque
import numpy as np
import cv2

class ToneStats40:
    def __init__(self,tiles):self.tiles=tiles;self.history=deque(maxlen=9)
    def update(self,image,initial=False):
        if initial:self.history.clear()
        h,w=image.shape;n=self.tiles
        cells=image.reshape(n,h//n,n,w//n).transpose(0,2,1,3).reshape(n*n,-1)
        low,median,high=np.quantile(cells,[.2,.5,.8],axis=1).reshape(3,n,n)
        spread=high-low;self.history.append((median.copy(),spread.copy()))
        refmedian=np.median(np.stack([a for a,_ in self.history]),axis=0)
        refspread=np.median(np.stack([a for _,a in self.history]),axis=0)
        gain=np.where((refspread<4)|(spread<4),1.,np.clip(refspread/np.maximum(spread,4),.85,1.15))
        shift=np.clip(refmedian-gain*median,-8,8)
        return cv2.resize(gain.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR),cv2.resize(shift.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)

def apply(image,fields,strength):
    gain,shift=fields
    return (image+strength*((gain-1)*image+shift)).clip(0,255)
