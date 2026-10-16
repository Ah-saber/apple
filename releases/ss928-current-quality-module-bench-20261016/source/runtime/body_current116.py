"""Learn current/plain-reference repair on confirmed large bipolar RAW tracks."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from compact_match93 import TOP,BOTTOM,CACHE92
from prepare_confidence76 import CACHE76
from prepare_gate80 import CACHE80
from physical_foreground95 import PHYS95
from prepare_plain115 import CACHE115
from mixed_body111 import MixedMotion111
from matched_positive104 import MatchedPositive104
from road_residual53 import support53

CACHE116=Path('/tmp/rawir-body116-20261014')
CACHE107=Path('/tmp/rawir-residual107-20261014')
OUT116=ROOT/'MOTION116-20261014'/'weather_heavy_c32'

class LargeTracks116:
    def __init__(self,road):
        self.road=road;self.medium=MixedMotion111(road);self.weak=MatchedPositive104(road)
    def step(self,presence,instant,noise,gate,physical,gain):
        matched=self.medium.step(presence,instant,noise)
        raw=cv2.GaussianBlur(presence.astype(np.float32),(0,0),.65)
        now=cv2.GaussianBlur(instant.astype(np.float32),(0,0),.65)
        seed=(np.abs(raw)>2.5*noise)&(raw*now>0)&(np.abs(now)>1.5*noise)&(self.road>.5)
        current=np.zeros_like(gate,np.uint8);past=current.copy();strong=[]
        for r in self.medium.last_matches:
            y,x,h,w=r['y'],r['x'],r['h'],r['w'];component=matched[y:y+h,x:x+w]>0
            selected=seed[y:y+h,x:x+w]&component
            positive=int(np.sum(selected&(raw[y:y+h,x:x+w]>0)))
            negative=int(np.sum(selected&(raw[y:y+h,x:x+w]<0)))
            if r['seeds']<64 or min(positive,negative)<8:continue
            current[y:y+h,x:x+w]|=component.astype(np.uint8)
            for lag,dy,dx in r['history']:
                past[y+dy:y+dy+h,x+dx:x+dx+w]|=component.astype(np.uint8)
            strong.append((r,component))
        self.weak.step(presence,gate,np.zeros_like(presence),physical,gain)
        protect=np.zeros_like(gate,np.uint8)
        for y,x,h,w,polarity,component,history in self.weak.last_matches:
            strength=float(np.mean(np.abs(gain*physical[y:y+h,x:x+w][component>0])))
            if strength>16:continue
            local=(component>0);inside=float(np.mean(current[y:y+h,x:x+w][local]))
            associated=False
            if inside>=.5:
                observed={lag:(dy,dx) for lag,dy,dx in history}
                for r,big in strong:
                    big_only=np.zeros_like(current);by,bx,bh,bw=r['y'],r['x'],r['h'],r['w']
                    big_only[by:by+bh,bx:bx+bw]=big
                    if float(np.mean(big_only[y:y+h,x:x+w][local]))<.5:continue
                    agreements=sum(lag in observed and max(abs(dy-observed[lag][0]),abs(dx-observed[lag][1]))<=1 for lag,dy,dx in r['history'])
                    if agreements>=2:associated=True;break
            if not associated:protect[y:y+h,x:x+w]|=local.astype(np.uint8)
        protect=cv2.dilate(protect,np.ones((13,13),np.uint8))
        current=cv2.dilate(current,np.ones((7,7),np.uint8))*(1-protect)
        past=cv2.dilate(past,np.ones((7,7),np.uint8))*(1-protect)
        mask=np.maximum(current,past).astype(np.float32)*self.road
        self.last_count=len(strong)
        return mask,current.astype(np.float32),past.astype(np.float32)

def hp(a,size=21):return a-cv2.blur(a,(size,size),borderType=cv2.BORDER_REFLECT)

def features116(source,corrected,plain,physical,foreground,presence,instant,current,past,road,gain):
    values=np.stack([source/255,corrected/255,plain/255,hp(source,9)/16,
        hp(corrected-source)/16,hp(plain-source)/16,gain*physical/16,foreground/16,
        presence/16,instant/16,current,past,road])
    return values.clip(-4,4).astype(np.float16)

class CurrentRepair116(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(13,32,3,padding=1,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=2,dilation=2,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4,padding_mode='reflect'),nn.ReLU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,mask):return 24*torch.tanh(self.net(features.clamp(-4,4)))*mask

def prepare():
    scene='weather_heavy_c32';gain=json.loads((ROOT/'MOTION95-20261014'/scene/'response.json').read_text())['gain']
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy();center=road>.9;center[:,:64]=False
    for split in ['train','val_continuous']:
        for path in sorted((DEST/scene/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text());out=CACHE116/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'features_complete.json').exists():continue
            raw=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r');instant=np.load(CACHE76/split/path.name/'instant_presence76.npy',mmap_mode='r')
            gate=np.load(CACHE80/split/path.name/'signed_gate80.npy',mmap_mode='r');noise=np.load(CACHE76/split/path.name/'noise76.npy')
            physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r');corrected=np.load(path/'current_only.npy',mmap_mode='r')
            plain=np.load(CACHE115/split/path.name/'plain_current.npy',mmap_mode='r')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r');parent=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            delta=np.load(CACHE107/split/path.name/'delta107.npy',mmap_mode='r')
            images=np.empty((len(source),BOTTOM-TOP,1280),np.float32)
            for i in range(len(source)):
                images[i]=np.clip(np.clip(source[i,TOP:BOTTOM].astype(np.float32)+parent[i,TOP:BOTTOM],0,255)+delta[i,TOP:BOTTOM],0,255)
            features=np.lib.format.open_memmap(out/'features116.npy',mode='w+',dtype=np.float16,shape=(len(source),13,BOTTOM-TOP,1280))
            masks=np.lib.format.open_memmap(out/'mask116.npy',mode='w+',dtype=np.float16,shape=images.shape)
            tracks=LargeTracks116(road);counts=[]
            for i,f in enumerate(meta['frames']):
                if i==0 or f!=meta['frames'][i-1]+1:tracks=LargeTracks116(road);sequence_start=i
                p=raw[i,TOP:BOTTOM].astype(np.float32);now=instant[i,TOP:BOTTOM].astype(np.float32)
                mask,current,past=tracks.step(p,now,float(noise[i,0]),gate[i,TOP:BOTTOM].astype(np.float32),physical[i],gain)
                if f<16:mask[:]=0
                background=np.median(images[max(sequence_start,i-64):i],axis=0) if i>sequence_start else images[i]
                fore=images[i]-background;fore-=np.median(fore[center])
                features[i]=features116(images[i],corrected[i,TOP:BOTTOM].astype(np.float32),plain[i,TOP:BOTTOM].astype(np.float32),
                    physical[i],fore,p,now,current,past,road,gain)
                masks[i]=mask;counts.append(dict(frame=f,components=tracks.last_count,pixels=int((mask>0).sum())))
                if i%20==0:print('BODY116_FEATURES',split,path.name,f,counts[-1],flush=True)
            features.flush();masks.flush();(out/'features_complete.json').write_text(json.dumps(dict(frames=meta['frames'],GT_used=False,feedback=False,counts=counts)))

if __name__=='__main__':
    cv2.setNumThreads(2);prepare()
