"""Recover all observations of confirmed training GT tracks, including their starts."""
import json
from collections import deque
import cv2
import numpy as np
import torch
from PIL import Image
from run_motion import ROOT
from target_mask_capture import CaptureMetric
from target_residual import ROOT as RUN_ROOT
from gap_metrics_v4 import dilate

def prepare(scene):
    out=RUN_ROOT/('allframe_targets_'+scene)
    if (out/'labels.pt').exists(): return torch.load(out/'labels.pt',map_location='cpu',weights_only=False)
    old=torch.load(RUN_ROOT/('guided_labels_'+scene)/'labels.pt',map_location='cpu',weights_only=False)
    row=old['sample']; assert row['split']=='train'
    roi=old['roi']; y,x,h,w=roi
    paths=sorted((ROOT/'data'/row['target']['path']).parent.glob('*.png'))
    images={int(p.stem):np.asarray(Image.open(p),dtype=np.float32) for p in paths}
    metric=CaptureMetric(roi); metric.prepare_background(list(images.values()))
    for f,image in images.items(): metric.update(f,image)
    valid={id(item) for item in metric.tracker.selected()}
    regions={f:[] for f in images}; all_masks={f:np.zeros((h,w),np.int8) for f in images}
    for region in metric.captured:
        item=region['item']; f=item['frame']; target=region['target_yx'].astype(np.int16)
        all_masks[f][target[:,0]-y,target[:,1]-x]=item['polarity']
        if id(item) not in valid: continue
        regions[f].append(dict(item=item,target=target,ring=region['ring_yx'].astype(np.int16)))
    labels={}; history=deque(maxlen=8); previous=None; last=None
    for f,image in images.items():
        if last is not None and f!=last+1: history.clear(); previous=None
        current=np.zeros((h,w),np.int8)
        for region in regions[f]:
            t=region['target']; current[t[:,0]-y,t[:,1]-x]=region['item']['polarity']
        departed=np.zeros_like(current); patch=image[y:y+h,x:x+w]
        if history and previous is not None:
            historical=np.zeros_like(current)
            for before in history: historical[before!=0]=before[before!=0]
            delta=patch-previous; delta-=np.median(delta)
            motion=np.abs(cv2.GaussianBlur(delta,(0,0),.7))>=np.maximum(3,2*np.pad(metric.noise,3,mode='edge'))
            departed[(historical!=0)&~dilate(all_masks[f]!=0,3)&~dilate(motion,2)]=historical[(historical!=0)&~dilate(all_masks[f]!=0,3)&~dilate(motion,2)]
        def sparse(a):
            yy,xx=np.nonzero(a)
            return np.stack([yy+y,xx+x,a[yy,xx]],1).astype(np.int16) if len(yy) else np.empty((0,3),np.int16)
        labels[f]=dict(target=sparse(current),ghost=sparse(departed))
        history.append(current); previous=patch; last=f
    selected=[r['item'] for rs in regions.values() for r in rs]
    summary=dict(scene=scene,frames=len(labels),regions=len(selected),tiny=sum(r['target_pixels']<=9 for r in selected),weak=sum(abs(r['gt_gap_gray'])<=15 for r in selected),tracks=metric.tracker.summary(),train_ROI=roi,GT_scope='training sequence, training ROI only; offline confirmed track observations; no future RAW inputs',previous_saved_frames=len(old['labels']))
    saved=dict(scene=scene,sample=row,roi=roi,labels=labels,regions=regions,summary=summary)
    out.mkdir(exist_ok=False); torch.save(saved,out/'labels.pt'); (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('ALLFRAME_TARGETS',json.dumps(summary),flush=True)
    return saved
