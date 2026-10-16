"""Isolate the frozen SR boundary by moving mirrored RAW into its fixed canvas."""
import gc
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from target_residual import ROOT,current_setup
from prepare_motion14 import DEST
from causal_detail import GuidanceView
from input_anchor12 import apply
from tieddecode_model11 import boards

CACHE91=Path('/tmp/rawir-residual91-20261014')
PAD91=32
WIDTH91=64

def shifted_box91(original,roi_width):
    shifted=original.clone()
    shifted[:,1]-=PAD91/roi_width
    shifted[:,3]-=PAD91/roi_width
    return shifted

def shifted_input91(inputs):
    return F.pad(inputs[...,:-PAD91],(PAD91,0,0,0),mode='reflect')

def mask91():
    return np.clip((WIDTH91-np.arange(WIDTH91,dtype=np.float32))/PAD91,0,1)[None,:]

def run():
    scene='weather_heavy_c32'
    kind,config,df,box,count,base,original,profile=current_setup(scene)
    del original
    for split,data_split in [('val_continuous','val')]:
        source=df(config,data_split);apply(source,scene)
        for path in sorted((DEST/scene/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text());row=meta['row']
            out=CACHE91/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():
                continue
            old=np.load(path/'current_only.npy',mmap_mode='r')
            # Training labels cannot choose an edge-only correction. It is zero there.
            result=np.lib.format.open_memmap(out/'delta91.npy',mode='w+',dtype=np.float32,shape=old.shape)
            result[:]=0
            reference_box=shifted_box91(box(row),row['eval_crop_tlhw'][3])
            parent,model=boards(scene,base,count,kind,reference_box,500);del parent
            data=GuidanceView(source,list(range(9-count,9)))
            for i,f in enumerate(meta['frames']):
                rr=dict(row,frame_id=f);data.current_record,data.current_crop,data.cache=rr,(0,0,1024,1280),{}
                context,_=data.context_for(rr);context=context[None].cuda().half()
                inputs=data.inputs(profile)[None].cuda()[:,-model.source_count:].half()
                current=shifted_input91(inputs[:,-1:].expand_as(inputs).contiguous())
                with torch.inference_mode():
                    full=model(current,context)
                    if tuple(full.shape)!=(1,1,3072,3840):
                        raise ValueError('Frozen complete SR geometry required')
                    shifted=full[0,0,::3,3*PAD91:3*(PAD91+WIDTH91):3].half().float().cpu().numpy()
                result[i,:,:WIDTH91]=np.clip(shifted-old[i,:,:WIDTH91].astype(np.float32),-8,8)*mask91()
                if i%20==0:
                    print('EDGE91',split,path.name,f,flush=True)
            result.flush()
            (out/'complete.json').write_text(json.dumps(dict(GT_used=False,weights_changed=False,
                shifted_reference_box=reference_box.tolist(),input_shift=PAD91,correction_width=WIDTH91,max_delta=8)))
            del model,data;gc.collect();torch.cuda.empty_cache()
    out=ROOT/'MOTION91-20261014'/scene;out.mkdir(parents=True,exist_ok=True)
    selected=dict(alpha=1.,eligible=None,reason='fixed boundary input treatment; no GT parameter fit')
    report=dict(scene=scene,parent_delta=77,parent_file='total77.npy',best=selected,diagnostic_best=selected,
                status='fixed_parameter',fit_split='none',validation_used_for_parameters=False,test_used=False,
                GT_used_for_parameters=False,training_ROI_affected=False,input_shift=PAD91,correction_width=WIDTH91)
    (out/'fit.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run()
