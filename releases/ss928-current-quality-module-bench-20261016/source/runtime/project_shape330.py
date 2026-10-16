"""Separate foreground shape redistribution from a component-wide tone change."""
import cv2,numpy as np
from foreground_support324 import foreground_support324

def project_shape330(delta,compact,active,road,small_only=False,conserve_mean=True):
    if small_only:
        weight=np.clip(compact,0,1)*(1-np.clip(active,0,1))*road
        delta=delta*weight
        if not conserve_mean:return delta
    else:weight=foreground_support324(compact,active)*road
    count,labels=cv2.connectedComponents((weight>0).astype(np.uint8),8)
    result=delta.copy()
    for k in range(1,count):
        region=labels==k;total=float(weight[region].sum())
        if total>0:result[region]-=weight[region]*(float(delta[region].sum())/total)
    return result
