"""Causal physical background and aligned foreground, with frozen local tone."""
from collections import deque
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from prepare_gate80 import CACHE80
from residual50 import RawSignal50
from road_residual53 import support53
from fit_residual50 import fit

TOP, BOTTOM = 176, 654
CACHE90 = Path('/tmp/rawir-residual90-20261014')
PHYSICAL90 = Path('/tmp/rawir-physical90-20261014')

def blur(a, size=21):
    return cv2.blur(a, (size, size), borderType=cv2.BORDER_REFLECT)

def detail(a):
    enhanced = a + .5 * np.clip(a-cv2.GaussianBlur(a, (0, 0), 1.1), -2, 2)
    return np.minimum(np.maximum(enhanced, cv2.erode(a, np.ones((3,3), np.uint8))),
                      cv2.dilate(a, np.ones((3,3), np.uint8)))

class FrozenTone90:
    """Existing stable-tone parameters; initialize once using past RAW only."""
    def __init__(self, reference):
        h, w = reference.shape
        self.low, self.high = np.percentile(reference[::2, ::2], [.1, 99.9])
        if self.high-self.low < 20:
            middle = (self.high+self.low)/2
            self.low, self.high = middle-10, middle+10
        bins = np.rint(np.clip((reference-self.low)/(self.high-self.low)*255, 0, 255)).astype(np.uint8)
        self.luts = np.empty((8,8,256), np.float32)
        th, tw = h//8, w//8
        for y in range(8):
            for x in range(8):
                hist = np.bincount(bins[y*th:(y+1)*th,x*tw:(x+1)*tw].ravel(), minlength=256).astype(np.float64)
                clipped = np.minimum(hist, 1.5*hist.sum()/256)
                clipped += (hist.sum()-clipped.sum())/256
                self.luts[y,x] = np.cumsum(clipped)*(255/hist.sum())
        cy = np.arange(TOP,BOTTOM,dtype=np.float32)/th-.5
        cx = np.arange(w,dtype=np.float32)/tw-.5
        y0, x0 = np.floor(cy).astype(int), np.floor(cx).astype(int)
        self.coords = (np.clip(y0,0,7)[:,None],np.clip(y0+1,0,7)[:,None],
                       np.clip(x0,0,7)[None,:],np.clip(x0+1,0,7)[None,:],
                       (cy-y0)[:,None],(cx-x0)[None,:])
    def apply(self, a):
        u = np.clip((a-self.low)/(self.high-self.low)*255,0,255).astype(np.float32)
        bins = np.rint(u).astype(np.uint8)
        y0,y1,x0,x1,fy,fx = self.coords
        lut = self.luts
        local = (1-fy)*((1-fx)*lut[y0,x0,bins]+fx*lut[y0,x1,bins])+fy*((1-fx)*lut[y1,x0,bins]+fx*lut[y1,x1,bins])
        # Keep floating gray until the video writer, avoiding intermediate rounding.
        return (3+249*(np.clip((.35*u+.65*local)/255,0,1)**1.12)).astype(np.float32)

def clean_common(a, reference):
    result = a.astype(np.float32).copy()
    result -= np.median(result-reference)
    difference = result-reference
    rows = np.clip(np.median(difference,axis=1),-3,3)
    columns = np.clip(np.median(difference,axis=0),-3,3)
    rows -= np.median(rows);columns -= np.median(columns)
    result -= rows[:,None]+columns[None,:]
    return result

def motion_planes(a):
    return [cv2.blur(a,k,borderType=cv2.BORDER_REFLECT) for k in [(1,7),(7,1),(3,3)]]

class PhysicalReference90:
    def __init__(self, initial):
        self.reference = np.median(initial,axis=0).astype(np.float32)
        self.tone = FrozenTone90(self.reference)
        self.reference = self.reference[TOP:BOTTOM]
        samples = [clean_common(a[TOP:BOTTOM],self.reference)-self.reference for a in initial]
        self.noise = []
        for k in range(3):
            planes = np.stack([motion_planes(a)[k] for a in samples])
            center = np.median(planes,axis=0)
            noise = 1.4826*np.median(np.abs(planes-center),axis=0)
            noise = cv2.GaussianBlur(noise,(0,0),3)
            self.noise.append(np.maximum(noise,max(.1,.6*float(np.median(noise)))))
        self.bank = deque(maxlen=4)
        self.yy,self.xx = np.mgrid[:BOTTOM-TOP,:initial.shape[2]].astype(np.float32)

    def step(self, current, background, signed_gate):
        background = clean_common(background,self.reference)
        current = clean_common(current,background)
        difference = current-background
        snr = np.maximum.reduce([np.abs(p)/n for p,n in zip(motion_planes(difference),self.noise)])
        observed = np.clip((snr-2.25)/1.5,0,1)
        observed = cv2.dilate(observed,np.ones((3,3),np.uint8))
        # Protected low-frequency field follows the existing weather primitive.
        delta = cv2.GaussianBlur(difference,(0,0),.7)
        weight = np.exp(-(delta/2)**4)*(1-observed)
        numerator = cv2.GaussianBlur(np.clip(difference,-3,3)*weight,(0,0),4)
        denominator = cv2.GaussianBlur(weight,(0,0),4)
        field = numerator/np.maximum(denominator,.05)
        field -= np.mean(field)
        current -= .9*field
        raw_fore = cv2.GaussianBlur(current-background,(0,0),.65)
        gray_background = self.tone.apply(detail(background))
        gray_fore = self.tone.apply(detail(current))-gray_background
        descriptor = np.clip(128+8*raw_fore,0,255).astype(np.float32)
        witness = np.clip((np.abs(signed_gate)-.15)/.85,0,1)
        total = gray_fore*observed
        total_weight = observed.copy()
        activity = observed.copy()
        for lag in [1,2,4]:
            if len(self.bank)<lag:
                continue
            old_descriptor,old_raw,old_gray,old_observed = self.bank[-lag]
            backward = cv2.calcOpticalFlowFarneback(descriptor,old_descriptor,None,.5,3,15,3,5,1.1,0)
            forward = cv2.calcOpticalFlowFarneback(old_descriptor,descriptor,None,.5,3,15,3,5,1.1,0)
            mx,my = self.xx+backward[...,0],self.yy+backward[...,1]
            def warp(a):
                return cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            inverse = warp(forward)
            old_raw_here = warp(old_raw)
            valid = ((np.linalg.norm(backward+inverse,axis=2)<.75+.05*np.linalg.norm(backward,axis=2)) &
                     (mx>=0)&(mx<descriptor.shape[1]-1)&(my>=0)&(my<descriptor.shape[0]-1)&
                     (raw_fore*old_raw_here>0)&(old_raw_here*signed_gate>0))
            old_mask = warp(old_observed)
            reliability = old_mask*valid*witness
            weight = reliability/(1+lag/4)
            total += warp(old_gray)*weight
            total_weight += weight
            activity = np.maximum(activity,reliability)
        fused = np.divide(total,total_weight,out=np.zeros_like(total),where=total_weight>1e-6)
        self.bank.append((descriptor.copy(),raw_fore.copy(),gray_fore.copy(),observed.copy()))
        return gray_background+activity*fused

def prepare():
    raw = RawSignal50('weather_heavy_c32')
    for path in sorted((DEST/'weather_heavy_c32'/'train').glob('capture_*')):
        meta = json.loads((path/'complete.json').read_text())
        raw.for_record(meta['row'],0,(TOP,0,BOTTOM-TOP,1280))
        physical = raw.open[meta['row']['raw']['path']]
        out = PHYSICAL90/path.name
        out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            continue
        frames = meta['frames']
        gate = np.load(CACHE80/'train'/path.name/'signed_gate80.npy',mmap_mode='r')
        desired = np.lib.format.open_memmap(out/'desired90.npy',mode='w+',dtype=np.float32,
                                          shape=(len(frames),BOTTOM-TOP,1280))
        desired[:] = 0
        reference = None
        for i,f in enumerate(frames):
            if f<16:
                continue
            if reference is None:
                reference = PhysicalReference90(np.asarray(physical[f-16:f],np.float32))
            background = np.median(np.asarray(physical[max(0,f-64):f,TOP:BOTTOM],np.float32),axis=0)
            desired[i] = reference.step(physical[f,TOP:BOTTOM].astype(np.float32),background,
                                        gate[i,TOP:BOTTOM].astype(np.float32))
            if i%10==0:
                print('PHYSICAL90',path.name,i,flush=True)
        desired.flush()
        (out/'complete.json').write_text(json.dumps(dict(frames=frames,raw_path=meta['row']['raw']['path'],
            GT_used=False,initialization='past16 RAW',background='past64 RAW',flow_lags=[1,2,4])))

def infer():
    road = support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta = json.loads((path/'complete.json').read_text())
            physical_meta = json.loads((PHYSICAL90/path.name/'complete.json').read_text())
            # The existing spatial split has identical physical input frames.
            if meta['row']['raw']['path']!=physical_meta['raw_path'] or meta['frames']!=physical_meta['frames']:
                raise ValueError('Physical input sequence differs from the cached causal reference')
            desired = np.load(PHYSICAL90/path.name/'desired90.npy',mmap_mode='r')
            source = np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            parent = np.load(Path('/tmp/rawir-residual77-20261014')/split/path.name/'total77.npy',mmap_mode='r')
            out = CACHE90/split/path.name;out.mkdir(parents=True,exist_ok=True)
            result = np.lib.format.open_memmap(out/'delta90.npy',mode='w+',dtype=np.float32,shape=source.shape)
            result[:] = 0
            for i,f in enumerate(meta['frames']):
                if f<16:
                    continue
                image = np.clip(source[i,TOP:BOTTOM].astype(np.float32)+parent[i,TOP:BOTTOM],0,255)
                difference = desired[i]-image
                result[i,TOP:BOTTOM] = np.clip(difference-blur(difference),-8,8)*road*min((f-15)/3,1)
            result.flush()
            (out/'complete.json').write_text(json.dumps(dict(GT_used_for_inference=False,parent=77,max_delta_gray=8,
                initialization='first16 unchanged, next3 fade',tone_parameters_fit=False)))
    fit(90,road=True,parent_delta=77,parent_file='total77.npy')

if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2)
    prepare();infer()
