"""Estimate slow drift on a reduced grid while retaining full-resolution compact-motion masks."""
import cv2
import numpy as np
import time
from streaming_cached_data import GuidanceView as BaseView, InputModel

def fast_correct(frame,reference,step):
    residual=frame-reference
    common=np.median(residual[::step,::step])
    row=np.median((residual-common)[:,::step],axis=1)[:,None]
    col=np.median((residual-common-row)[::step],axis=0)[None,:]
    local=residual-common-row-col
    high=local-cv2.GaussianBlur(local,(0,0),2)
    sample=high[::step,::step]
    center=float(np.median(sample))
    sigma=max(1e-5,1.4826*float(np.median(np.abs(sample-center))))
    seeds=(np.abs(high-center)>=3*sigma).astype(np.uint8)
    count,labels,boxes,_=cv2.connectedComponentsWithStats(seeds,8)
    mask=np.zeros(seeds.shape,np.uint8)
    for ident in range(1,count):
        x,y,w,h,area=boxes[ident]
        if 2<=area<=256 and w<=48 and h<=64:
            patch=mask[y:y+h,x:x+w]
            patch[labels[y:y+h,x:x+w]==ident]=1
    weight=(1-cv2.dilate(mask,np.ones((7,7),np.uint8))).astype(np.float32)
    shape=(max(1,frame.shape[1]//step),max(1,frame.shape[0]//step))
    down=lambda x:cv2.resize(x,shape,interpolation=cv2.INTER_AREA)
    blur=lambda x:cv2.GaussianBlur(x,(0,0),4/step)
    denominator=blur(down(weight))
    numerator=blur(down(local*weight))
    fallback=blur(down(local))
    field=np.divide(numerator,denominator,out=fallback.copy(),where=denominator>=.1)
    broad=cv2.resize(field,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(frame-common-row-col-.9*broad,dtype=np.float32)

class GuidanceView(BaseView):
    def __init__(self,source,indices):
        super().__init__(source,indices)
        self.control_preprocess_log=[]

    def normalized_stack(self,record,crop,operation=0,downsample=False):
        started=time.perf_counter()
        value=super().normalized_stack(record,crop,operation,downsample)
        self.control_preprocess_log.append(dict(frame=int(record['frame_id']),elapsed_ms=(time.perf_counter()-started)*1000))
        return value

    def correction(self,value,reference,pipeline):
        if not pipeline.startswith('streamCachedFast'):
            return super().correction(value,reference,pipeline)
        step=4 if 'Fast4' in pipeline else 2
        return fast_correct(value,reference,step)

