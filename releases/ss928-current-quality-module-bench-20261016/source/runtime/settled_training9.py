import math
import torch
from torch.nn import functional as F

def settle(values):
    before=after=removed=0
    axis=torch.arange(-2,3,dtype=torch.float32);g=torch.exp(-axis.square()/(2*.7**2));g/=g.sum();kernel=(g[:,None]*g[None,:])[None,None]
    for v in values:
        delta=(v['pair_gt'].float()-v['gt'].float())*255
        smooth=F.conv2d(F.pad(delta,(2,2,2,2),mode='reflect'),kernel)
        inner=smooth[...,12:-12,12:-12];drift=inner.median();centered=smooth-drift
        sigma=float((inner-drift).abs().median())/(.67448975*math.sqrt(2))
        motion=(centered.abs()>=max(3.,2*sigma)).float();v['gt_motion']=motion
        for prefix in ['', 'pair_']:
            ghost=v[prefix+'ghost'];before+=int((ghost!=0).sum())
            occupied=F.max_pool2d((v[prefix+'target'].abs()>0).float(),7,1,3)
            allowed=(1-occupied)*(1-F.max_pool2d(motion,5,1,2))
            current=ghost*allowed;after+=int((current!=0).sum());removed+=int(((ghost!=0)&(current==0)).sum())
            v[prefix+'ghost']=current
    return dict(before_nonzero_pixels=before,after_nonzero_pixels=after,excluded_nonzero_pixels=removed,
        current_GT_exclusion_radius=3,adjacent_GT_motion_exclusion_radius=2,
        GT_motion_source='Existing training GT pair only; sigma .7 smoothing, median drift removed, max(3,2sigma) threshold',
        masks='Signed settled departure; exclude current target vicinity and active GT changes; original caches unchanged')

def center_temporal(delta,static):
    shifts=[]
    for i in range(delta.shape[0]):
        x=delta[i,0,12:-12,12:-12];mask=static[i,0,12:-12,12:-12]>0
        selected=x[mask];shifts.append(selected.median() if selected.numel() else x.sum()*0)
    return delta-torch.stack(shifts)[:,None,None,None]
