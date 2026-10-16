"""Limit a new shape correction to current causal RAW ownership."""
import cv2,numpy as np

def foreground_support324(compact,active,small_only=False):
    large=np.zeros_like(active) if small_only else cv2.erode(active.astype(np.float32),np.ones((7,7),np.uint8))
    owner=(np.maximum(compact,large)>.15).astype(np.uint8)
    return np.clip(cv2.distanceTransform(owner,cv2.DIST_L2,3)/2,0,1)
