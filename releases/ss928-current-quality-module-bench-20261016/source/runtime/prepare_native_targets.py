"""Add training-only native-pixel tiny tracks; keep evaluation masks unchanged."""
import copy
import inspect
import json
import textwrap
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import gap_metrics_v4
from gt_tracks_v5 import GTTracks
from target_mask_capture import CaptureMetric
from prepare_allframe_targets import prepare as existing_prepare
from target_residual import ROOT
from run_motion import ROOT as DATA_ROOT

_source=textwrap.dedent(inspect.getsource(gap_metrics_v4.GapMetrics.update))
_source=_source.replace('cv2.GaussianBlur(dg, (0, 0), .7)','dg').replace('cv2.GaussianBlur(g, (0, 0), .7)','g')
_source=_source.replace('2 <= area <= 256','1 <= area <= 256')
_source=_source.replace(' < 2',' < 1')
_source=_source.replace('frame_targets.append(item)','self.capture_mask(item,target,ring,top,left)\n        frame_targets.append(item)')
_namespace=dict(vars(gap_metrics_v4)); exec(compile(_source,'native_training_capture','exec'),_namespace)
_update=_namespace['update']

class PixelTracks(GTTracks):
    def sample(self,image,point):
        y,x=point; y=int(np.clip(round(y-self.origin[0]),0,image.shape[0]-1)); x=int(np.clip(round(x-self.origin[1]),0,image.shape[1]-1))
        return float(image[y,x]),(y,x)

class NativeCapture(CaptureMetric):
    def __init__(self,crop):
        super().__init__(crop); self.tracker=PixelTracks(); y,x,_,_=self.crop; self.tracker.origin=(y+3,x+3)
    def prepare_background(self,frames):
        y,x,h,w=self.crop; values=np.stack([g[y+3:y+h-3,x+3:x+w-3] for g in frames]).astype(np.float32)
        initial=np.median(values,axis=0)
        for g in values:g-=np.median(g-initial)
        self.background=np.median(values,axis=0); self.noise=np.maximum(1,1.4826*np.median(np.abs(values-self.background),axis=0)).astype(np.float32)
    def update(self,frame,gt,predictions=None):
        y,x,h,w=self.crop; self.tracker.images[frame]=gt[y+3:y+h-3,x+3:x+w-3]; self.tracker.noise=self.noise
        _update(self,frame,gt,{'GT':gt})

def prepare(scene):
    out=ROOT/('native_targets_'+scene)
    if (out/'labels.pt').exists(): return torch.load(out/'labels.pt',map_location='cpu',weights_only=False)
    saved=copy.deepcopy(existing_prepare(scene)); row=saved['sample']; assert row['split']=='train'
    paths=sorted((DATA_ROOT/'data'/row['target']['path']).parent.glob('*.png'))
    images={int(p.stem):np.asarray(Image.open(p),dtype=np.float32) for p in paths if int(p.stem) in saved['labels']}
    metric=NativeCapture(saved['roi']); metric.prepare_background(list(images.values()))
    for f,image in images.items():metric.update(f,image)
    valid={id(item) for item in metric.tracker.selected()}; added=0; added_pixels=0
    for region in metric.captured:
        item=region['item']; f=item['frame']; t=region['target_yx'].astype(np.int16)
        if id(item) not in valid or len(t)>9:continue
        old=saved['labels'][f]['target']; oldset={tuple(p) for p in old[:,:2]}
        if any(tuple(p) in oldset for p in t):continue
        saved['regions'][f].append(dict(item=item,target=t,ring=region['ring_yx'].astype(np.int16)))
        signs=np.full((len(t),1),item['polarity'],np.int16)
        saved['labels'][f]['target']=np.concatenate([old,np.concatenate([t,signs],1)])
        ghosts=saved['labels'][f]['ghost']; occupied={tuple(p) for p in t}
        saved['labels'][f]['ghost']=np.asarray([p for p in ghosts if tuple(p[:2]) not in occupied],np.int16).reshape(-1,3)
        added+=1; added_pixels+=len(t)
    saved['summary'].update(native_tiny_observations_added=added,native_tiny_pixels_added=added_pixels,native_track_rule='unblurred training GT; same appearance and polarity; >=5 observations; displacement>=2; departure and arrival >=max(3gray,.35local gap,1.5 native GT noise)',evaluation_rules_changed=False,original_training_labels_retained=True)
    out.mkdir(exist_ok=False); torch.save(saved,out/'labels.pt'); (out/'summary.json').write_text(json.dumps(saved['summary'],indent=2))
    print('NATIVE_TRAIN_LABELS',scene,json.dumps(saved['summary']),flush=True)
    return saved
