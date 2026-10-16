"""Restore only missing foreground after causal compact RAW motion matching."""
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
from compact_match93 import TOP,BOTTOM,CACHE92,components93
from physical_foreground95 import PHYS95
from fixed_background98 import CACHE98
from road_residual53 import support53
from fit_residual50 import fit
from repair_motion15 import load_captures

CACHE104=Path('/tmp/rawir-residual104-20261014')

class MatchedPositive104:
    def __init__(self,road):self.road=road;self.bank=deque(maxlen=4)
    def step(self,presence,gate,foreground,physical_fore,gain):
        raw=cv2.GaussianBlur(presence.astype(np.float32),(0,0),.65)
        regions,_=components93(gate,self.road);correction=np.zeros_like(foreground)
        self.last_matches=[]
        for y,x,h,w,polarity,mask in regions:
            y0,x0=max(0,y-4),max(0,x-4)
            y1,x1=min(raw.shape[0],y+h+4),min(raw.shape[1],x+w+4)
            template=raw[y0:y1,x0:x1];matches=0;moved=False;observed_matches=[]
            for lag in [1,2,4]:
                if len(self.bank)<lag:continue
                old_raw,old_gate=self.bank[-lag]
                sy0,sx0=max(0,y0-3),max(0,x0-3)
                sy1,sx1=min(raw.shape[0],y1+3),min(raw.shape[1],x1+3)
                response=cv2.matchTemplate(old_raw[sy0:sy1,sx0:sx1],template,cv2.TM_CCOEFF_NORMED)
                yy,xx=np.mgrid[:response.shape[0],:response.shape[1]]
                dy_grid,dx_grid=yy+sy0-y0,xx+sx0-x0
                score=response-.015*(dy_grid*dy_grid+dx_grid*dx_grid)
                iy,ix=np.unravel_index(np.argmax(score),score.shape)
                if float(response[iy,ix])<.6:continue
                dy,dx=int(dy_grid[iy,ix]),int(dx_grid[iy,ix])
                previous=old_gate[y+dy:y+dy+h,x+dx:x+dx+w]
                if float(np.mean((polarity*previous[mask>0])>.15))<.25:continue
                matches+=1;moved|=abs(dy)+abs(dx)>=1
                observed_matches.append((lag,dy,dx))
            if matches<2 or not moved:continue
            self.last_matches.append((y,x,h,w,polarity,mask,observed_matches))
            difference=gain*physical_fore[y:y+h,x:x+w]-foreground[y:y+h,x:x+w]
            amount=polarity*np.clip(polarity*difference,0,8)
            valid=physical_fore[y:y+h,x:x+w]*polarity>0
            correction[y:y+h,x:x+w]=amount*mask*np.abs(gate[y:y+h,x:x+w])*valid
        self.bank.append((raw.copy(),gate.copy()))
        return correction*self.road

def prepare():
    gain=json.loads((ROOT/'MOTION95-20261014'/'weather_heavy_c32'/'response.json').read_text())['gain']
    alpha98=json.loads((ROOT/'MOTION98-20261014'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    center=road>.9;center[:,:64]=False
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text())
            physical_meta=json.loads((PHYS95/path.name/'complete.json').read_text())
            if meta['frames']!=physical_meta['frames'] or meta['row']['raw']['path']!=physical_meta['raw_path']:
                raise ValueError('Physical sequence differs')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            parent=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            background_delta=np.load(CACHE98/split/path.name/'delta98.npy',mmap_mode='r')
            images=source[:,TOP:BOTTOM].astype(np.float32)+parent[:,TOP:BOTTOM]
            physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r')
            gate=np.load(CACHE80/split/path.name/'signed_gate80.npy',mmap_mode='r')
            presence=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r')
            controller=MatchedPositive104(road)
            out=CACHE104/split/path.name;out.mkdir(parents=True,exist_ok=True)
            result=np.lib.format.open_memmap(out/'delta104.npy',mode='w+',dtype=np.float32,shape=source.shape);result[:]=0
            for i,f in enumerate(meta['frames']):
                result[i]=alpha98*background_delta[i]
                background=np.median(images[max(0,i-64):i],axis=0) if i else images[0]
                foreground=images[i]-background;foreground-=np.median(foreground[center])
                proposal=controller.step(presence[i,TOP:BOTTOM].astype(np.float32),
                    gate[i,TOP:BOTTOM].astype(np.float32),foreground,physical[i],gain)
                if f>=16:result[i,TOP:BOTTOM]+=proposal
            result.flush();(out/'complete.json').write_text(json.dumps(dict(parent=92,background98_alpha=alpha98,
                gain_fit='original training ROI95',history='independent RAW lag1/2/4',current_confirmation='compact93',
                matches_required=2,moving_match_required=True,GT_used_for_inference=False,feedback=False)))

def select104():
    fit(104,road=True,parent_delta=92,parent_file='total92.npy')
    path=ROOT/'MOTION104-20261014'/'weather_heavy_c32'/'fit.json'
    report=json.loads(path.read_text());settings=[r['alpha'] for r in report['candidates']]
    body={a:[] for a in settings};tiny={a:[] for a in settings}
    for c in load_captures('weather_heavy_c32',raw_targets=True):
        cap='capture_%02d'%c['meta']['capture']
        labels=torch.load(ROOT/'BODY_LABEL83-20261014'/cap/'labels.pt',map_location='cpu',weights_only=False)
        source=np.load(CACHE50/'train'/cap/'source43.npy',mmap_mode='r')
        parent=np.load(CACHE92/'train'/cap/'total92.npy',mmap_mode='r')
        delta=np.load(CACHE104/'train'/cap/'delta104.npy',mmap_mode='r')
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
    path.write_text(json.dumps(report,indent=2));print('COMBINE104_FIT',json.dumps(report),flush=True)




if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2);prepare();select104()

