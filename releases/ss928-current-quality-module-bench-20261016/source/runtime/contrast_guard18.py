"""Limit temporal detail loss and departure excess; keep the stable coarse field."""
import json,argparse,itertools
import numpy as np
import torch
import cv2
from target_residual import ROOT
from prepare_motion14 import DEST
from repair_gate17 import Repair17
from fit_motion14 import collect,measure

OUT18=ROOT/'MOTION18-20261013'

def guard(stable,current,activity,p):
    current_local=current-cv2.blur(current,(7,7),borderType=cv2.BORDER_REFLECT)
    stable_local=stable-cv2.blur(stable,(7,7),borderType=cv2.BORDER_REFLECT)
    active=(activity>p['activity']) & (np.maximum(np.abs(current_local),np.abs(stable_local))>p['minimum'])
    bounded=np.clip(stable_local,current_local-p['tolerance'],current_local+p['tolerance'])
    return (stable+active*(bounded-stable_local)).clip(0,255)

def prepare(scene):
    ck=torch.load(ROOT/'MOTION17-20261013'/scene/'last_step500.pt',map_location='cpu',weights_only=False)
    model=Repair17().cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'])
    for path in sorted((DEST/scene/'train').glob('capture_*')):
        if (path/'repair17-complete.json').exists():continue
        meta=json.loads((path/'complete.json').read_text());frames=meta['frames']
        arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['baseline','current_only','gated_history','past_background','activity','occupancy']}
        saved={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype=np.float32,shape=arrays['baseline'].shape)
               for k in ['repair17','current_repair15']}
        previous=None;previous_output=None
        with torch.inference_mode():
            for i,f in enumerate(frames):
                v={k:torch.from_numpy(np.array(a[i],dtype=np.float32))[None,None].cuda() for k,a in arrays.items()}
                first=i==0 or f!=frames[i-1]+1
                pred,_,_,corrected=model(v,v if first else previous,v['baseline'] if first else previous_output,initial=first)
                saved['repair17'][i]=pred[0,0].cpu().numpy().clip(0,255)
                saved['current_repair15'][i]=corrected[0,0].cpu().numpy().clip(0,255)
                previous,previous_output=v,pred
        for a in saved.values():a.flush()
        (path/'repair17-complete.json').write_text(json.dumps(dict(scene=scene,split='train',steps=0)))
        print('MOTION18_TRAIN_FIELDS',scene,meta['capture'],flush=True)

def fit(scene):
    captures=collect(scene)
    for c,path in zip(captures,sorted((DEST/scene/'train').glob('capture_*'))):
        indices=c['indices'];frames=len(c['frames']);c['parents']={};local={}
        for name in ['repair15','repair17','current_repair15']:
            array=np.load(path/(name+'.npy'),mmap_mode='r')
            c['parents'][name]=np.asarray(array.reshape(frames,-1)[:,indices],dtype=np.float32)
            local[name]=np.stack([(a.astype(np.float32)-cv2.blur(a.astype(np.float32),(7,7),borderType=cv2.BORDER_REFLECT)).ravel()[indices] for a in array])
        c['local']=local
    original=measure(captures,dict(gate=0.,gain=0.,beta=0.));rows=[]
    for parent,activity,tolerance,minimum in itertools.product(['repair15','repair17'],[.1,.25,.5,.75],[.25,.5,1.,2.],[.5,1.,2.]):
        p=dict(parent=parent,activity=activity,tolerance=tolerance,minimum=minimum);values=[]
        for c in captures:
            stable=c['parents'][parent];cl=c['local']['current_repair15'];sl=c['local'][parent]
            active=(c['values']['activity']>activity)&(np.maximum(np.abs(cl),np.abs(sl))>minimum)
            bounded=np.clip(sl,cl-tolerance,cl+tolerance)
            pred=(stable+active*(bounded-sl)).clip(0,255)
            cc=dict(c);cc['values']=dict(c['values'],baseline=pred);values.append(cc)
        metrics=measure(values,dict(gate=0.,gain=0.,beta=0.))
        score=sum(weight*metrics[k]/max(original[k],.05) for k,weight in [('gap',2),('ghost',1),('static',1),('photo',.2)] if original[k] is not None)
        score+=5*max(0,metrics['inversions']-original['inversions'])/max(original['targets'],1)
        rows.append(dict(parameters=p,metrics=metrics,score=score))
    eligible=[r for r in rows if r['metrics']['inversions']<=original['inversions'] and
              all(original[k] is None or r['metrics'][k]<=original[k] for k in ['gap','ghost','static'])]
    best=min(eligible or rows,key=lambda r:r['score'])
    report=dict(scene=scene,original=original,best=best,all_train_objectives_nonworse=bool(eligible),
                fit_split='train',validation_used_for_fit=False,test_used=False,new_optimizer_steps=0,
                ranked=sorted(rows,key=lambda r:r['score']))
    (OUT18/(scene+'-fit.json')).write_text(json.dumps(report,indent=2))
    print('MOTION18_FIT',json.dumps({k:report[k] for k in ['scene','original','best','all_train_objectives_nonworse']}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
    OUT18.mkdir(exist_ok=True);prepare(a.scene);fit(a.scene)
