"""Joint current-target and nearby background repair with one tone reference."""
from pathlib import Path
import numpy as np,torch
from small_temporal258 import SmallTemporal258
CACHE279=Path('/tmp/rawir-joint279-20261016')
CACHE280=Path('/tmp/rawir-joint280-20261016')
CACHE282=Path('/tmp/rawir-joint282-20261016')
class JointContext279(SmallTemporal258):
    def forward(self,features,mask):return 32*torch.tanh(self.net(features.clamp(-4,4)))*mask
def joint_mask279(compact,release,large,include_large=False):
    import cv2
    core=np.maximum(compact,release)
    if include_large:core=np.maximum(core,large)
    distance=cv2.distanceTransform((core<=.1).astype(np.uint8),cv2.DIST_C,3)
    mask=np.clip((24-distance)/8,0,1);mask[:,:64]=0
    return mask.astype(np.float32)
def joint_raw279(small,large):
    return np.where(np.any(np.abs(small)>0,axis=0)[None],small,large).astype(np.float16)
