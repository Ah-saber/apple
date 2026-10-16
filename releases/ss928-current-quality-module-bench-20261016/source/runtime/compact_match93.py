"""Causal compact RAW matching to fill weak foreground and clear departures."""
from collections import deque
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from prepare_confidence76 import CACHE76
from prepare_gate80 import CACHE80
from road_residual53 import support53
from fit_residual50 import fit

TOP,BOTTOM=176,654
CACHE92=Path('/tmp/rawir-residual92-20261014')
CACHE93=Path('/tmp/rawir-residual93-20261014')

def components93(gate,road):
    regions=[];confirmed=np.zeros_like(gate,np.float32)
    for polarity in [-1,1]:
        seeds=((polarity*gate>.5)&(road>.5)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(seeds,connectivity=8)
        for index in range(1,count):
            x,y,w,h,area=stats[index]
            if not 2<=area<=128 or max(w,h)>16:
                continue
            mask=(labels[y:y+h,x:x+w]==index).astype(np.float32)
            confirmed[y:y+h,x:x+w][mask>0]=polarity
            regions.append((int(y),int(x),int(h),int(w),polarity,mask))
    return regions,confirmed

class CompactMatch93:
    def __init__(self):
        self.road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
        self.center_mask=self.road>.9
        self.center_mask[:,:64]=False
        self.bank=deque(maxlen=4)
    def step(self,presence,current,gate,foreground,raw_noise,sr_noise):
        raw=cv2.GaussianBlur(presence.astype(np.float32),(0,0),.65)
        regions,confirmed=components93(gate,self.road)
        correction=np.zeros_like(foreground)
        for y,x,h,w,polarity,mask in regions:
            y0,x0=max(0,y-4),max(0,x-4)
            y1,x1=min(raw.shape[0],y+h+4),min(raw.shape[1],x+w+4)
            template=raw[y0:y1,x0:x1]
            total=foreground[y:y+h,x:x+w]*.5
            weights=np.full((h,w),.5,np.float32)
            for lag in [1,2,4]:
                if len(self.bank)<lag:continue
                old_raw,old_gate,old_fore,_=self.bank[-lag]
                sy0,sx0=max(0,y0-3),max(0,x0-3)
                sy1,sx1=min(raw.shape[0],y1+3),min(raw.shape[1],x1+3)
                response=cv2.matchTemplate(old_raw[sy0:sy1,sx0:sx1],template,cv2.TM_CCOEFF_NORMED)
                dy_grid,dx_grid=np.mgrid[:response.shape[0],:response.shape[1]]
                dy_grid=dy_grid+sy0-y0;dx_grid=dx_grid+sx0-x0
                score=response-.015*(dy_grid*dy_grid+dx_grid*dx_grid)
                iy,ix=np.unravel_index(np.argmax(score),score.shape)
                correlation=float(response[iy,ix])
                if correlation<.6:continue
                dy,dx=int(dy_grid[iy,ix]),int(dx_grid[iy,ix])
                previous_raw=old_raw[y+dy:y+dy+h,x+dx:x+dx+w]
                previous_fore=old_fore[y+dy:y+dy+h,x+dx:x+dx+w]
                previous_gate=old_gate[y+dy:y+dy+h,x+dx:x+dx+w]
                current_amplitude=float(np.mean(polarity*raw[y:y+h,x:x+w][mask>0]))
                old_amplitude=float(np.mean(polarity*previous_raw[mask>0]))
                ratio=np.clip(current_amplitude/max(old_amplitude,raw_noise),.5,2.)
                valid=(polarity*previous_gate>.15)&(polarity*previous_fore>0)
                weight=np.clip((correlation-.6)/.4,0,1)*valid/(1+lag/4)
                total+=previous_fore*ratio*weight;weights+=weight
            desired=total/weights
            # Use the observed component pixels, not its bounding-box background.
            amount=np.clip(polarity*(desired-foreground[y:y+h,x:x+w]),0,4)
            correction[y:y+h,x:x+w]=polarity*amount*mask*np.abs(gate[y:y+h,x:x+w])
        if len(self.bank)>=2:
            old1,old2=self.bank[-1][3],self.bank[-2][3]
            old_direction=np.sign(old1)
            overlap=(old1*old2>0)
            quiet=(np.abs(gate)<.15)&(np.abs(presence)<raw_noise)&(np.abs(current)<raw_noise)
            retained=foreground*old_direction>sr_noise
            old_sr_confirmed=(self.bank[-1][2]*old_direction>sr_noise)&(self.bank[-2][2]*old_direction>sr_noise)
            departure=overlap&quiet&retained&old_sr_confirmed
            correction[departure]=-old_direction[departure]*np.minimum(np.abs(foreground[departure]),4)
        self.bank.append((raw.copy(),gate.copy(),foreground.copy(),confirmed.copy()))
        return correction*self.road

def prepare_parent92(split,path):
    source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
    parent=np.load(Path('/tmp/rawir-residual77-20261014')/split/path.name/'total77.npy',mmap_mode='r')
    body=np.load(Path('/tmp/rawir-residual82-20261014')/split/path.name/'delta82.npy',mmap_mode='r')
    edge=np.load(Path('/tmp/rawir-residual91-20261014')/split/path.name/'delta91.npy',mmap_mode='r') if split=='val_continuous' else None
    alpha=json.loads((ROOT/'MOTION82-20261014'/'weather_heavy_c32'/'fit.json').read_text())['best']['alpha']
    out=CACHE92/split/path.name;out.mkdir(parents=True,exist_ok=True)
    total=np.lib.format.open_memmap(out/'total92.npy',mode='w+',dtype=np.float32,shape=source.shape)
    images=np.empty((len(source),BOTTOM-TOP,1280),np.float32)
    for i in range(len(source)):
        value=np.clip(source[i].astype(np.float32)+parent[i],0,255)
        value=np.clip(value+alpha*body[i],0,255)
        if edge is not None:value=np.clip(value+edge[i],0,255)
        total[i]=value-source[i].astype(np.float32)
        images[i]=value[TOP:BOTTOM]
    total.flush()
    # Edge91 is zero in the original training ROI and in its local match windows.
    return source.shape,images

def run():
    scene='weather_heavy_c32'
    for split in ['train','val_continuous']:
        for path in sorted((DEST/scene/split).glob('capture_*')):
            out=CACHE93/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            meta=json.loads((path/'complete.json').read_text())
            shape,images=prepare_parent92(split,path)
            p=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r')
            now=np.load(CACHE76/split/path.name/'instant_presence76.npy',mmap_mode='r')
            noise=np.load(CACHE76/split/path.name/'noise76.npy')
            gates=np.load(CACHE80/split/path.name/'signed_gate80.npy',mmap_mode='r')
            result=np.lib.format.open_memmap(out/'delta93.npy',mode='w+',dtype=np.float32,shape=shape)
            result[:]=0;matcher=CompactMatch93()
            for i,f in enumerate(meta['frames']):
                background=np.median(images[max(0,i-64):i] if i else images[:1],axis=0)
                foreground=images[i]-background
                foreground-=np.median(foreground[matcher.center_mask])
                correction=matcher.step(p[i,TOP:BOTTOM],now[i,TOP:BOTTOM],gates[i,TOP:BOTTOM].astype(np.float32),
                                        foreground,float(noise[i,0]),float(noise[i,1]))
                if i>=16:result[i,TOP:BOTTOM]=correction
                if i%20==0:print('COMPACT93',split,path.name,f,flush=True)
            result.flush()
            (out/'complete.json').write_text(json.dumps(dict(GT_used_for_inference=False,parent=92,history=[1,2,4],
                historical_output='independent92; never corrected93',first16_unchanged=True,max_delta_gray=4)))
            del images
    fit(93,road=True,parent_delta=92,parent_file='total92.npy')

if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2);run()
