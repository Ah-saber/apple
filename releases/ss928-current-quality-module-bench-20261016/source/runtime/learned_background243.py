"""Learn background repair and carry each foreground component with its local ring."""
from pathlib import Path
import torch
import cv2,numpy as np
from torch.nn import functional as F
from small_context227 import SmallContext227

CACHE243=Path('/tmp/rawir-background243-20261016')
CACHE247=Path('/tmp/rawir-background247-20261016')

def build_masks243(source,raw,owner,current,protect,active,support,road,whole_owner=False):
    core=road>.9;core[:,:64]=False;values=raw[core]
    sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
    confirmed=np.maximum(np.maximum(owner,current),protect)
    foreground=cv2.dilate(((confirmed>0)&(np.abs(raw)>2*sigma)).astype(np.uint8),np.ones((3,3),np.uint8))
    if whole_owner:foreground=cv2.dilate((owner>0).astype(np.uint8),np.ones((13,13),np.uint8))
    tiny=(protect>0)&(owner<=0)
    _,labels,boxes,_=cv2.connectedComponentsWithStats((active>0).astype(np.uint8),8)
    selected=np.zeros_like(foreground)
    for k in range(1,len(boxes)):
        if boxes[k,cv2.CC_STAT_AREA]>=128:selected[labels==k]=1
    distance=cv2.distanceTransform((selected==0).astype(np.uint8),cv2.DIST_C,3)
    route=np.clip((24-distance)/12,0,1)
    weight=(np.ones_like(support) if whole_owner else support)*route*road*(~tiny)
    if whole_owner:weight[foreground>0]=(route*(~tiny))[foreground>0]
    weight[:,:64]=0
    _,labels=cv2.connectedComponents(foreground,connectivity=8)
    valid=((foreground==0)&(~tiny)&(road>.9)).astype(np.float32)
    return np.stack([weight,labels.astype(np.float32),source,valid]).astype(np.float32)

class LearnedBackground243(SmallContext227):
    def background(self,features,masks):
        valid=masks[:,3:4]
        raw=6*torch.tanh(self.net(features.clamp(-4,4)))
        blur=lambda a:F.avg_pool2d(F.pad(a,(2,2,2,2),mode='reflect'),5,1)
        return blur(raw*valid)/blur(valid).clamp_min(1e-3)*masks[:,0:1]*valid

    def project(self,delta,masks):
        outputs=[]
        for b in range(len(delta)):
            image=masks[b,2];labels=masks[b,1];valid=masks[b,3]>0
            result=delta[b,0]
            for label in torch.unique(labels).tolist():
                if label<=0:continue
                target=labels==label
                if not (masks[b,0][target]>0).any():continue
                shape=target.float()[None,None]
                outer=F.max_pool2d(shape,13,1,6)[0,0]>0
                inner=F.max_pool2d(shape,5,1,2)[0,0]>0
                ring=outer&(~inner)&valid
                if int(ring.sum())<12:continue
                offset=torch.quantile((image+delta[b,0])[ring],.5)-torch.quantile(image[ring],.5)
                result=torch.where(target,offset*masks[b,0],result)
            outputs.append(result[None])
        return torch.stack(outputs)

    def forward(self,features,masks):return self.project(self.background(features,masks),masks)
