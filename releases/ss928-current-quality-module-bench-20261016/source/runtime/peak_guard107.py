"""Limit isolated spatial contrast peaks with independently observed matched history."""
import json
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from prepare_confidence76 import CACHE76
from prepare_gate80 import CACHE80
from compact_match93 import TOP,BOTTOM,CACHE92
from physical_foreground95 import PHYS95
from fixed_background98 import CACHE98
from matched_positive104 import MatchedPositive104
from road_residual53 import support53
from fit_residual50 import fit
from repair_motion15 import load_captures

CACHE107=Path('/tmp/rawir-residual107-20261014')

class PeakGuard107:
    def __init__(self,road):self.road=road;self.bank=deque(maxlen=4)
    def step(self,current,base,positive,matches,gate):
        reduction=np.zeros_like(positive)
        height,width=current.shape
        for y,x,h,w,polarity,mask,history in matches:
            y0,x0=max(0,y-8),max(0,x-8)
            y1,x1=min(height,y+h+8),min(width,x+w+8)
            component=np.zeros((y1-y0,x1-x0),np.uint8)
            component[y-y0:y-y0+h,x-x0:x-x0+w]=mask.astype(np.uint8)
            ring=(cv2.dilate(component,np.ones((13,13),np.uint8))>0)&~(cv2.dilate(component,np.ones((5,5),np.uint8))>0)
            ring&=(np.abs(gate[y0:y1,x0:x1])<.15)&(self.road[y0:y1,x0:x1]>.5)
            if ring.sum()<12:continue
            t=np.argwhere(component>0)+[y0,x0];r=np.argwhere(ring)+[y0,x0]
            ty,tx=t[:,0],t[:,1];ry,rx=r[:,0],r[:,1]
            actual=polarity*float(current[ty,tx].mean()-np.median(current[ry,rx]))
            floor=polarity*float(base[ty,tx].mean()-np.median(base[ry,rx]))
            previous=[]
            for lag,dy,dx in history:
                if len(self.bank)<lag:continue
                old=self.bank[-lag]
                valid=(ry+dy>=0)&(ry+dy<height)&(rx+dx>=0)&(rx+dx<width)
                if valid.sum()<12:continue
                if np.any(ty+dy<0) or np.any(ty+dy>=height) or np.any(tx+dx<0) or np.any(tx+dx>=width):continue
                previous.append(polarity*float(old[ty+dy,tx+dx].mean()-np.median(old[ry[valid]+dy,rx[valid]+dx])))
            if not previous:continue
            ceiling=max(floor,max(previous))+.5
            excess=max(0.,actual-ceiling)
            amplitude=polarity*float(positive[ty,tx].mean())
            scale=np.clip(excess/max(amplitude,1e-6),0,1)
            reduction[y:y+h,x:x+w]=positive[y:y+h,x:x+w]*mask*scale
        # Preserve the independent unguarded104 image, never the guarded result.
        self.bank.append(current.copy())
        return reduction

def prepare():
    gain=json.loads((ROOT/'MOTION95-20261014'/'weather_heavy_c32'/'response.json').read_text())['gain']
    alpha98=json.loads((ROOT/'MOTION98-20261014'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
    alpha104=json.loads((ROOT/'MOTION104-20261014'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    center=road>.9;center[:,:64]=False
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text())
            pm=json.loads((PHYS95/path.name/'complete.json').read_text())
            if pm['frames']!=meta['frames'] or pm['raw_path']!=meta['row']['raw']['path']:
                raise ValueError('Physical sequence differs')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            total=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            background_delta=np.load(CACHE98/split/path.name/'delta98.npy',mmap_mode='r')
            images=source[:,TOP:BOTTOM].astype(np.float32)+total[:,TOP:BOTTOM]
            physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r')
            gate=np.load(CACHE80/split/path.name/'signed_gate80.npy',mmap_mode='r')
            presence=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r')
            matcher=MatchedPositive104(road);guard=PeakGuard107(road)
            out=CACHE107/split/path.name;out.mkdir(parents=True,exist_ok=True)
            result=np.lib.format.open_memmap(out/'delta107.npy',mode='w+',dtype=np.float32,shape=source.shape);result[:]=0
            for i,f in enumerate(meta['frames']):
                background=np.median(images[max(0,i-64):i],axis=0) if i else images[0]
                foreground=images[i]-background;foreground-=np.median(foreground[center])
                signed=gate[i,TOP:BOTTOM].astype(np.float32)
                positive=alpha104*matcher.step(presence[i,TOP:BOTTOM].astype(np.float32),signed,foreground,physical[i],gain)
                base_unclipped=images[i]+alpha104*alpha98*background_delta[i,TOP:BOTTOM]
                base=np.clip(base_unclipped,0,255)
                independent=np.clip(base_unclipped+positive,0,255)
                reduction=guard.step(independent,base,positive,matcher.last_matches,signed)
                result[i]=alpha104*alpha98*background_delta[i]
                if f>=16:result[i,TOP:BOTTOM]+=positive-reduction
            result.flush();(out/'complete.json').write_text(json.dumps(dict(parent=92,components=dict(background98=alpha98,foreground104=alpha104),
                history='independent unguarded104, current RAW matching',GT_used_for_inference=False,
                ceiling_margin_gray=.5,feedback=False,foreground_reduction_only=True)))

def select107():
    fit(107,road=True,parent_delta=92,parent_file='total92.npy')
    path=ROOT/'MOTION107-20261014'/'weather_heavy_c32'/'fit.json'
    report=json.loads(path.read_text());settings=[r['alpha'] for r in report['candidates']]
    body={a:[] for a in settings};tiny={a:[] for a in settings}
    for c in load_captures('weather_heavy_c32',raw_targets=True):
        cap='capture_%02d'%c['meta']['capture']
        labels=torch.load(ROOT/'BODY_LABEL83-20261014'/cap/'labels.pt',map_location='cpu',weights_only=False)
        source=np.load(CACHE50/'train'/cap/'source43.npy',mmap_mode='r')
        parent=np.load(CACHE92/'train'/cap/'total92.npy',mmap_mode='r')
        delta=np.load(CACHE107/'train'/cap/'delta107.npy',mmap_mode='r')
        for i,f in enumerate(c['meta']['frames']):
            if i not in c['valid']:continue
            image=np.clip(source[i].astype(np.float32)+parent[i],0,255);gt=c['arrays']['GT'][i]
            for region in labels['regions'].get(f,[]):
                t=np.asarray(region['target'],int);ring=np.asarray(region['ring'],int)
                truth=float(gt[t[:,0],t[:,1]].astype(np.float32).mean()-np.median(gt[ring[:,0],ring[:,1]].astype(np.float32)))
                for alpha in settings:
                    target=np.clip(image[t[:,0],t[:,1]]+alpha*delta[i,t[:,0],t[:,1]],0,255)
                    back=np.clip(image[ring[:,0],ring[:,1]]+alpha*delta[i,ring[:,0],ring[:,1]],0,255)
                    error=abs(float(target.mean()-np.median(back))-truth)
                    body[alpha].append(error)
                    if len(t)<=16:tiny[alpha].append(error)
    whole={a:float(np.mean(v)) for a,v in body.items()};small={a:float(np.mean(v)) for a,v in tiny.items()}
    report['strict_selection']=report['best'];zero=report['candidates'][0]
    for row in report['candidates']:
        alpha=row['alpha'];row['body_gap_error']=whole[alpha];row['tiny_body_gap_error']=small[alpha]
        row['strict_eligible']=row['eligible']
        row['eligible']=(all(row['metrics'][k]<=report['source'][k]+.05 for k in ['gap','ghost','signed_ghost']) and
                         all(q<=1.005*p for q,p in zip(row['capture_flicker'],zero['capture_flicker'])) and
                         whole[alpha]<=whole[0.]+1e-6 and small[alpha]<=small[0.]+1e-6)
        row['score']+=4*whole[alpha]/max(whole[0.],.05)+6*small[alpha]/max(small[0.],.05)
    report['best']=min([r for r in report['candidates'] if r['eligible']],key=lambda r:r['score'])
    report['diagnostic_best']=min(report['candidates'],key=lambda r:r['score'])
    report.update(status='passed' if report['best']['alpha']>0 else 'no_pass',body_source_gap_error=whole[0.],
                  tiny_body_source_gap_error=small[0.],body_labels_reference='BODY_LABEL83 original train ROI',
                  tolerance_non_flicker_gray=.05,validation_used=False,test_used=False)
    path.write_text(json.dumps(report,indent=2));print('COMBINE107_FIT',json.dumps(report),flush=True)




if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2);prepare();select107()

