"""Fit two signed body features inside independently matched RAW components."""
import json
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from prepare_confidence76 import CACHE76
from prepare_gate80 import CACHE80
from compact_match93 import TOP,BOTTOM,CACHE92
from physical_foreground95 import PHYS95
from road_residual53 import support53
from repair_motion15 import load_captures

CACHE111=Path('/tmp/rawir-residual111-20261014')
CACHE107=Path('/tmp/rawir-residual107-20261014')

class MixedMotion111:
    def __init__(self,road):self.road=road;self.bank=deque(maxlen=4)
    def step(self,presence,instant,raw_noise):
        raw=cv2.GaussianBlur(presence.astype(np.float32),(0,0),.65)
        now=cv2.GaussianBlur(instant.astype(np.float32),(0,0),.65)
        seed=(np.abs(raw)>2.5*raw_noise)&(raw*now>0)&(np.abs(now)>1.5*raw_noise)&(self.road>.5)
        connected=cv2.dilate(seed.astype(np.uint8),np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(connected,connectivity=8)
        mask=np.zeros_like(presence,np.float32);self.last_matches=[]
        for k in range(1,n):
            x,y,w,h,_=stats[k];component=labels[y:y+h,x:x+w]==k
            seeds=int(np.sum(seed[y:y+h,x:x+w]&component))
            if not 17<=seeds<=2048 or max(h,w)>64:continue
            y0,x0=max(0,y-4),max(0,x-4);y1,x1=min(raw.shape[0],y+h+4),min(raw.shape[1],x+w+4)
            template=raw[y0:y1,x0:x1];matches=[];moved=False
            for lag in [1,2,4]:
                if len(self.bank)<lag:continue
                old,old_seed=self.bank[-lag]
                sy0,sx0=max(0,y0-4),max(0,x0-4);sy1,sx1=min(raw.shape[0],y1+4),min(raw.shape[1],x1+4)
                response=cv2.matchTemplate(old[sy0:sy1,sx0:sx1],template,cv2.TM_CCOEFF_NORMED)
                yy,xx=np.mgrid[:response.shape[0],:response.shape[1]]
                dy_grid,dx_grid=yy+sy0-y0,xx+sx0-x0
                score=response-.015*(dy_grid*dy_grid+dx_grid*dx_grid)
                iy,ix=np.unravel_index(np.argmax(score),score.shape)
                if float(response[iy,ix])<.6:continue
                dy,dx=int(dy_grid[iy,ix]),int(dx_grid[iy,ix])
                previous=old_seed[y+dy:y+dy+h,x+dx:x+dx+w]
                if float(np.mean(previous[component]))<.25:continue
                matches.append((lag,dy,dx));moved|=abs(dy)+abs(dx)>=1
            if len(matches)<2 or not moved:continue
            mask[y:y+h,x:x+w]=np.maximum(mask[y:y+h,x:x+w],component)
            self.last_matches.append(dict(y=int(y),x=int(x),h=int(h),w=int(w),seeds=seeds,history=matches,component=component.copy()))
        self.bank.append((raw.copy(),seed.copy()))
        return mask*self.road

def hp(a):return a-cv2.blur(a,(21,21),borderType=cv2.BORDER_REFLECT)

def prepare_features():
    gain=json.loads((ROOT/'MOTION95-20261014'/'weather_heavy_c32'/'response.json').read_text())['gain']
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy();center=road>.9;center[:,:64]=False
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text());out=CACHE111/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'features_complete.json').exists():continue
            physical_meta=json.loads((PHYS95/path.name/'complete.json').read_text())
            if physical_meta['frames']!=meta['frames'] or physical_meta['raw_path']!=meta['row']['raw']['path']:raise ValueError('Physical sequence differs')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r');parent=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            previous_delta=np.load(CACHE107/split/path.name/'delta107.npy',mmap_mode='r')
            images=np.empty((len(source),BOTTOM-TOP,1280),np.float32)
            for i in range(len(source)):
                images[i]=np.clip(np.clip(source[i,TOP:BOTTOM].astype(np.float32)+parent[i,TOP:BOTTOM],0,255)+previous_delta[i,TOP:BOTTOM],0,255)
            physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r')
            presence=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r');single=np.load(path/'current_only.npy',mmap_mode='r')
            instant=np.load(CACHE76/split/path.name/'instant_presence76.npy',mmap_mode='r');noise=np.load(CACHE76/split/path.name/'noise76.npy')
            features=np.lib.format.open_memmap(out/'features111.npy',mode='w+',dtype=np.float16,shape=(len(source),2,BOTTOM-TOP,1280))
            controller=MixedMotion111(road);counts=[]
            for i,f in enumerate(meta['frames']):
                mask=controller.step(presence[i,TOP:BOTTOM].astype(np.float32),instant[i,TOP:BOTTOM].astype(np.float32),float(noise[i,0]))
                features[i]=0
                if f>=16:
                    background=np.median(images[max(0,i-64):i],axis=0) if i else images[0]
                    foreground=images[i]-background;foreground-=np.median(foreground[center])
                    features[i,0]=np.clip(gain*physical[i]-foreground,-8,8)*mask
                    features[i,1]=np.clip(hp(single[i,TOP:BOTTOM].astype(np.float32)-images[i]),-8,8)*mask
                counts.append(dict(frame=f,components=len(controller.last_matches),pixels=int((mask>0).sum())))
                if i%20==0:print('MIXED111_FEATURE',split,path.name,f,counts[-1],flush=True)
            features.flush();(out/'features_complete.json').write_text(json.dumps(dict(frames=meta['frames'],GT_used=False,feedback=False,counts=counts)))

def fit_response():
    gram=np.zeros((2,2),np.float64);rhs=np.zeros(2,np.float64);pixels=0;counts=[]
    for c in load_captures('weather_heavy_c32',raw_targets=True):
        cap='capture_%02d'%c['meta']['capture'];roi=c['saved']['roi'];y,x,h,w=roi
        source=np.load(CACHE50/'train'/cap/'source43.npy',mmap_mode='r');parent=np.load(CACHE92/'train'/cap/'total92.npy',mmap_mode='r')
        delta=np.load(CACHE107/'train'/cap/'delta107.npy',mmap_mode='r');features=np.load(CACHE111/'train'/cap/'features111.npy',mmap_mode='r');count=0
        y0,y1=max(y,TOP),min(y+h,BOTTOM)
        for i,f in enumerate(c['meta']['frames']):
            if i not in c['valid'] or f<16:continue
            values=features[i,:,y0-TOP:y1-TOP,x:x+w].astype(np.float64)
            valid=np.any(values!=0,axis=0)
            if not valid.any():continue
            image=np.clip(np.clip(source[i,y0:y1,x:x+w].astype(np.float32)+parent[i,y0:y1,x:x+w],0,255)+delta[i,y0:y1,x:x+w],0,255)
            target=np.clip(c['arrays']['GT'][i,y0:y1,x:x+w].astype(np.float32)-image,-8,8)
            a=values[:,valid].T;b=target[valid].astype(np.float64)
            gram+=a.T@a;rhs+=a.T@b;pixels+=len(b);count+=len(b)
        counts.append(dict(capture=c['meta']['capture'],roi=roi,pixels=count))
    if not pixels:raise ValueError('No matched original training pixels')
    coefficients=np.clip(np.linalg.solve(gram+.01*pixels*np.eye(2),rhs),-1,1)
    report=dict(coefficients=coefficients.tolist(),pixels=pixels,training=counts,fit_split='train',validation_used=False,test_used=False,ridge=.01,
        feature_limit=8,correction_limit=6,physical_gain_source=95,source=107)
    out=ROOT/'MOTION111-20261014'/'weather_heavy_c32';out.mkdir(parents=True,exist_ok=True)
    (out/'response.json').write_text(json.dumps(report,indent=2));print('MIXED111_RESPONSE',json.dumps(report),flush=True)
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            previous=np.load(CACHE107/split/path.name/'delta107.npy',mmap_mode='r');features=np.load(CACHE111/split/path.name/'features111.npy',mmap_mode='r')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r');parent=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            out=CACHE111/split/path.name;result=np.lib.format.open_memmap(out/'head111.npy',mode='w+',dtype=np.float32,shape=previous.shape)
            total=np.lib.format.open_memmap(out/'parent107.npy',mode='w+',dtype=np.float32,shape=previous.shape)
            for i in range(len(result)):
                total[i]=np.clip(np.clip(source[i].astype(np.float32)+parent[i],0,255)+previous[i],0,255)-source[i].astype(np.float32)
                result[i]=0;result[i,TOP:BOTTOM]=np.clip(np.einsum('c,cyx->yx',coefficients,features[i].astype(np.float32)),-6,6)
            result.flush();total.flush();(out/'complete.json').write_text(json.dumps(dict(parent=92,body_parent=107,GT_used_for_inference=False,feedback=False)))

if __name__=='__main__':
    cv2.setNumThreads(2);prepare_features();fit_response()

