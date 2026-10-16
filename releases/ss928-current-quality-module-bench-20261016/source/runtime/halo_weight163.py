"""Extend large-body repair while retaining original weights and tiny guards."""
import cv2,numpy as np

def grow163(weight,protect,road):
    core=weight>0
    if not np.any(core):return weight.copy()
    distance=cv2.distanceTransform((~core).astype(np.uint8),cv2.DIST_C,3)
    propagated=cv2.dilate(weight,np.ones((25,25),np.uint8))
    grown=propagated*np.clip((12-distance)/6,0,1)
    added=~core
    grown[added]=np.minimum(grown[added]*(1-protect[added]),road[added])
    grown[:,:64]=0;grown[core]=weight[core]
    return grown.astype(np.float16).astype(np.float32)
