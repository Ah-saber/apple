"""Train and freeze a causal physical/SR background detail correction."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from compact_match93 import TOP,BOTTOM,CACHE92
from residual50 import RawSignal50
from physical_reference90 import FrozenTone90,detail
from road_residual53 import support53
from fit_residual50 import fit

CACHE98=Path('/tmp/rawir-residual98-20261014')

def hp(a):return a-cv2.blur(a,(21,21),borderType=cv2.BORDER_REFLECT)

def features98(physical,sr):
    s=sr.astype(np.float32).copy();s[:,:3]=s[:,3:6][:,::-1]
    return np.stack([hp(physical-s),hp((physical-cv2.GaussianBlur(physical,(0,0),1.1))-
                        (s-cv2.GaussianBlur(s,(0,0),1.1)))])

def reference98(meta,images,raw):
    frames=meta['frames'];i=next(k for k,f in enumerate(frames) if f>=16)
    f=frames[i]
    if frames[:i]!=list(range(f-i,f)):raise ValueError('Continuous causal initialization required')
    raw.for_record(meta['row'],f,(TOP,0,BOTTOM-TOP,1280))
    physical=raw.open[meta['row']['raw']['path']]
    initial=np.median(np.asarray(physical[f-16:f],np.float32),axis=0)
    value=FrozenTone90(initial).apply(detail(initial[TOP:BOTTOM]))
    sr=np.median(images[:i],axis=0)
    return value,sr,i

def calibrate():
    covariance=np.zeros((2,2),np.float64);cross=np.zeros(2,np.float64);counts=[]
    raw=RawSignal50('weather_heavy_c32')
    road=support53((TOP,256,BOTTOM-TOP,768),'cpu')[0,0].numpy()>.9
    for path in sorted((DEST/'weather_heavy_c32'/'train').glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text())
        source=np.load(CACHE50/'train'/path.name/'source43.npy',mmap_mode='r')
        total=np.load(CACHE92/'train'/path.name/'total92.npy',mmap_mode='r')
        images=source[:,TOP:BOTTOM].astype(np.float32)+total[:,TOP:BOTTOM]
        physical,sr,i=reference98(meta,images,raw)
        gt=np.load(path/'GT.npy',mmap_mode='r')[:i,TOP:BOTTOM,256:1024].astype(np.float32)
        stable=gt-np.median(gt[:,road],axis=1)[:,None,None]
        mask=road&(np.std(stable,axis=0)<3)
        mask[:12]=False;mask[-12:]=False;mask[:,:12]=False;mask[:,-12:]=False
        features=features98(physical,sr)[:,:,256:1024][:,mask].T.astype(np.float64)
        truth=hp(np.median(gt,axis=0)-sr[:,256:1024])[mask].astype(np.float64)
        covariance+=features.T@features;cross+=features.T@truth;counts.append(int(mask.sum()))
    coefficients=np.clip(np.linalg.solve(covariance+np.eye(2)*max(1.,np.trace(covariance)*1e-5),cross),-2,2)
    report=dict(coefficients=coefficients.tolist(),train_pixels=counts,fit_split='train',
        GT_ROI=[0,256,1024,768],GT_outside_train_ROI_used=False,validation_used=False,test_used=False,
        initialization='available past16 median; static physical and independent92 backgrounds',
        features=['HP21(physical_background-SR_background)','HP21(detail1.1_physical-detail1.1_SR)'])
    out=ROOT/'MOTION98-20261014'/'weather_heavy_c32';out.mkdir(parents=True,exist_ok=True)
    (out/'background.json').write_text(json.dumps(report,indent=2));print('FIXED_BG98',json.dumps(report),flush=True)
    return coefficients.astype(np.float32)

def infer(coefficients):
    raw=RawSignal50('weather_heavy_c32');road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text())
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            total=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            images=source[:,TOP:BOTTOM].astype(np.float32)+total[:,TOP:BOTTOM]
            physical,sr,start=reference98(meta,images,raw)
            fixed=np.clip(np.einsum('c,cyx->yx',coefficients,features98(physical,sr)),-6,6)*road
            out=CACHE98/split/path.name;out.mkdir(parents=True,exist_ok=True)
            result=np.lib.format.open_memmap(out/'delta98.npy',mode='w+',dtype=np.float32,shape=source.shape);result[:]=0
            for i in range(start,len(source)):result[i,TOP:BOTTOM]=fixed*min((i-start+1)/3,1)
            result.flush();(out/'complete.json').write_text(json.dumps(dict(GT_used_for_inference=False,parent=92,
                fixed_after_initialization=True,background_feedback=False,initialization='past16',max_delta=6)))

if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2)
    infer(calibrate());fit(98,road=True,parent_delta=92,parent_file='total92.npy')
