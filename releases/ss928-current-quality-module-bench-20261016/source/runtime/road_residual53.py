"""C32 moving-road-target correction; retain the accepted background."""
import numpy as np
import torch
from residual50 import Residual50

def support53(crop,device='cuda'):
    y,x,h,w=crop
    xs=np.arange(x,x+w,dtype=np.float32)
    center=np.interp(xs,[0,96,256,512,768,1024,1279],[244,294,346,405,475,535,608]).astype(np.float32)
    distance=np.abs(np.arange(y,y+h,dtype=np.float32)[:,None]-center[None])
    return torch.from_numpy(np.clip((32-distance)/8,0,1))[None,None].to(device)

def on_road53(center):
    y,x=center
    line=np.interp(x,[0,96,256,512,768,1024,1279],[244,294,346,405,475,535,608])
    return abs(y-line)<24

class RoadResidual53(Residual50):
    def forward(self,v,source,previous_source):
        delta=super().forward(v,source,previous_source)
        crop=v.get('crop53',(0,0,*source.shape[-2:]))
        return delta*support53(crop,source.device)

def focus53(c):
    """Use physical RAW road regions and their nine strongest train GT pixels."""
    indices={f:i for i,f in enumerate(c['meta']['frames'])};regions={};road=[]
    for f,items in c['saved']['regions'].items():
        selected=[]
        for r in items:
            if not r.get('raw_confirmed_foreground'):
                selected.append(r);continue
            t=np.asarray(r['target'],int)
            if not on_road53(t.mean(0)):continue
            ring=np.asarray(r['ring'],int);gt=c['arrays']['GT'][indices[f]].astype(np.float32)
            contrast=gt[t[:,0],t[:,1]]-np.median(gt[ring[:,0],ring[:,1]])
            polarity=1 if np.mean(contrast)>=0 else -1
            t=t[np.argsort(polarity*contrast)[-min(9,len(t)):]]
            q=dict(r,target=t,road53=True);selected.append(q)
            if indices[f] in c['valid']:road.append((indices[f],q))
        regions[f]=selected
    c['saved']['regions']=regions;c['road53']=road
    return c
