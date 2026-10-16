"""Compute label-free physical presence and noise for each causal frame."""
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from prepare_motion14 import DEST
from residual50 import RawSignal50
from body_gain76 import history_fields76
from road_residual53 import support53

CACHE76=Path('/tmp/rawir-body76-inputs-20261014')
def run():
    scene='weather_heavy_c32';raw=RawSignal50(scene);top,bottom=176,654;crop=(top,0,bottom-top,1280)
    mask=support53(crop,'cpu')[0,0].numpy()>.9
    def sigma(a):return max(.2,float(1.4826*np.median(np.abs(a[mask]-np.median(a[mask])))))
    def hp(a):return a-cv2.blur(a,(21,21),borderType=cv2.BORDER_REFLECT)
    for split in ['train','val_continuous']:
        for path in sorted((DEST/scene/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text());row=meta['row'];out=CACHE76/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            current=np.load(path/'current_only.npy',mmap_mode='r');shape=current.shape
            presence=np.lib.format.open_memmap(out/'presence76.npy',mode='w+',dtype=np.float16,shape=shape)
            instantaneous=np.lib.format.open_memmap(out/'instant_presence76.npy',mode='w+',dtype=np.float16,shape=shape)
            coherent=np.lib.format.open_memmap(out/'coherent76.npy',mode='w+',dtype=np.uint8,shape=shape)
            noises=[]
            for i,f in enumerate(meta['frames']):
                features=raw.for_record(row,f,crop)[0].cpu().numpy();physical=raw.open[row['raw']['path']];scale=255/raw.scale[row['raw']['path']]
                ids=np.arange(max(0,f-64),f) if f else np.array([0])
                background=np.median(np.array(physical[ids,top:bottom],dtype=np.float32),axis=0)
                recent=np.maximum(np.arange(f-2,f+1),0)
                p=hp((np.mean(np.array(physical[recent,top:bottom],dtype=np.float32),axis=0)-background)*scale)
                now=hp((physical[f,top:bottom].astype(np.float32)-background)*scale)
                raw_noise=sigma(p)
                history=history_fields76(current,i,crop,np.zeros_like(p),device='cpu',repair_boundary=True)[0].numpy()
                sr_noise=sigma(history[1]*16)
                smoothed_presence=cv2.GaussianBlur(p,(0,0),.65)
                changes=np.maximum(np.abs(smoothed_presence),np.abs(cv2.GaussianBlur(features[12]*16,(0,0),.65)))
                smooth_noise=sigma(smoothed_presence)
                n,labels,_,_=cv2.connectedComponentsWithStats((changes>smooth_noise).astype(np.uint8),connectivity=8)
                counts=np.bincount(labels[changes>1.5*smooth_noise],minlength=n)
                keep=(counts[labels]>=2)&(labels!=0);keep=cv2.dilate(keep.astype(np.uint8),np.ones((17,17),np.uint8))
                presence[i]=0;presence[i,top:bottom]=p
                instantaneous[i]=0;instantaneous[i,top:bottom]=now
                coherent[i]=0;coherent[i,top:bottom]=keep
                noises.append([raw_noise,sr_noise])
                if i%20==0:print('BODY76_INPUT',split,path.name,f,flush=True)
            presence.flush();instantaneous.flush();coherent.flush();np.save(out/'noise76.npy',np.asarray(noises,np.float32))
            (out/'complete.json').write_text(json.dumps(dict(frames=len(shape) and shape[0],GT_used=False,history='available past up to64, no duplicated startup background',RAW_current_mean=3)))
if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory);run()
