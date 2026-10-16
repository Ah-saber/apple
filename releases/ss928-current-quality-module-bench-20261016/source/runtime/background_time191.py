"""Denoise background with two compatible past RAW observations."""
import cv2,numpy as np

def filter191(current,history,confirmed,road):
    core=road>.9;core[:,:64]=False
    values=current[core]
    sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
    total=current.copy();count=np.ones_like(current)
    for previous in history:
        weight=np.clip(1-np.abs(previous-current)/(2*sigma),0,1)
        total+=weight*previous;count+=weight
    outside=cv2.distanceTransform((confirmed<=0).astype(np.uint8),cv2.DIST_C,3)
    outside=np.clip(outside/3,0,1)
    return current+outside*(total/count-current),sigma
