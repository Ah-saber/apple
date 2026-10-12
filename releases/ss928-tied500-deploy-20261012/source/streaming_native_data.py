"""CPU native sampled medians, mask lookup and output; preserve floating-point order."""
import ctypes
from pathlib import Path
import cv2
import numpy as np
from streaming_fast_data import GuidanceView as BaseView,InputModel

LIB=ctypes.CDLL(str(Path(__file__).with_name('native_input_cpu.so')))
F=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
B=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
LIB.sampled_medians.argtypes=[F,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,F,F]
LIB.component_mask.argtypes=[I,B,B,ctypes.c_int]
LIB.corrected_output.argtypes=[F,F,F,F,ctypes.c_float,F,ctypes.c_int,ctypes.c_int]
for name in ['sampled_medians','component_mask','corrected_output']: getattr(LIB,name).restype=None

def native_correct(frame,reference,step):
    residual=frame-reference
    common=np.median(residual[::step,::step])
    height,width=frame.shape
    row=np.empty((height,1),np.float32); col=np.empty((1,width),np.float32)
    LIB.sampled_medians(residual,height,width,step,float(common),row,col)
    local=residual
    np.subtract(local,common,out=local); np.subtract(local,row,out=local); np.subtract(local,col,out=local)
    high=cv2.GaussianBlur(local,(0,0),2); np.subtract(local,high,out=high)
    sample=high[::step,::step]
    center=float(np.median(sample)); sigma=max(1e-5,1.4826*float(np.median(np.abs(sample-center))))
    np.subtract(high,center,out=high); np.abs(high,out=high)
    seeds=(high>=3*sigma).astype(np.uint8)
    _,labels,boxes,_=cv2.connectedComponentsWithStats(seeds,8)
    valid=((boxes[:,cv2.CC_STAT_AREA]>=2)&(boxes[:,cv2.CC_STAT_AREA]<=256)&
           (boxes[:,cv2.CC_STAT_WIDTH]<=48)&(boxes[:,cv2.CC_STAT_HEIGHT]<=64)).astype(np.uint8)
    valid[0]=0; mask=np.empty(seeds.shape,np.uint8)
    LIB.component_mask(labels,valid,mask,mask.size)
    weight=cv2.dilate(mask,np.ones((7,7),np.uint8)); np.subtract(1,weight,out=weight); weight=weight.astype(np.float32)
    shape=(max(1,width//step),max(1,height//step))
    down=lambda x:cv2.resize(x,shape,interpolation=cv2.INTER_AREA)
    blur=lambda x:cv2.GaussianBlur(x,(0,0),4/step)
    denominator=blur(down(weight)); numerator=blur(down(local*weight)); field=blur(down(local))
    np.divide(numerator,denominator,out=field,where=denominator>=.1)
    broad=cv2.resize(field,(width,height),interpolation=cv2.INTER_LINEAR)
    output=np.empty_like(frame)
    LIB.corrected_output(frame,row,col,broad,float(common),output,height,width)
    return output

class GuidanceView(BaseView):
    def correction(self,value,reference,pipeline):
        if pipeline.startswith('streamCachedFast'):
            return native_correct(value,reference,4 if 'Fast4' in pipeline else 2)
        return super().correction(value,reference,pipeline)
