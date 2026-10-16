"""Fit only on original training ROIs and existing train target labels."""
import sys, json, argparse, itertools
from pathlib import Path
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST, REPORT

def load_labels(scene):
    path = ROOT / ('ALLCAPTUREFIX13-'+scene+'.json')
    if path.exists():
        index = json.loads(path.read_text())
        return {(c['domain'], c['sequence']): torch.load(c['labels_path'], map_location='cpu', weights_only=False)
                for c in index['captures']}
    saved = torch.load(ROOT / ('short_train_tracks11_'+scene) / 'labels.pt', map_location='cpu', weights_only=False)
    return {(saved['sample']['domain'], saved['sample']['sequence_id']): saved}

def collect(scene):
    captures=[]; labels=load_labels(scene)
    for path in sorted((DEST/scene/'train').glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text())
        identity=tuple(meta['identity']); saved=labels.get(identity)
        if saved is None: raise ValueError('Missing original capture labels '+str(identity))
        if saved['sample']['split']!='train': raise ValueError('Train labels only')
        y,x,h,w=saved['roi']; frames=meta['frames']
        grid_y,grid_x=np.mgrid[y+8:y+h-8:8,x+8:x+w-8:8]
        static_indices=(grid_y*1280+grid_x).ravel()
        chosen=set(int(i) for i in static_indices); region_specs=[]; ghosts=[]
        for f in frames:
            rs=[]
            for r in saved['regions'].get(f,[]):
                if not 1<=len(r['target'])<=9: continue
                t=np.asarray(r['target'],dtype=int); ring=np.asarray(r['ring'],dtype=int)
                ti=t[:,0]*1280+t[:,1]; ri=ring[:,0]*1280+ring[:,1]
                chosen.update(map(int,ti)); chosen.update(map(int,ri)); rs.append((ti,ri))
            region_specs.append(rs)
            ghost=np.asarray(saved['labels'].get(f,{}).get('ghost',np.empty((0,3))),dtype=int).reshape(-1,3)
            gi=ghost[:,0]*1280+ghost[:,1] if len(ghost) else np.empty(0,dtype=int)
            chosen.update(map(int,gi)); ghosts.append((gi,ghost[:,2] if len(ghost) else np.empty(0)))
        indices=np.array(sorted(chosen),dtype=int); index_map={int(k):i for i,k in enumerate(indices)}
        select=lambda a: np.array([index_map[int(k)] for k in a],dtype=int)
        values={}
        for name in ['baseline','current_only','gated_history','past_background','activity','occupancy','GT']:
            a=np.load(path/(name+'.npy'),mmap_mode='r')
            values[name]=np.asarray(a.reshape(len(frames),-1)[:,indices],dtype=np.float32)
        regions=[[(select(t),select(r)) for t,r in rr] for rr in region_specs]
        ghosts=[(select(g),s) for g,s in ghosts]
        captures.append(dict(values=values,regions=regions,ghosts=ghosts,static=select(static_indices),
                             frames=frames,identity=identity,roi=saved['roi'],indices=indices))
    return captures

def predict(v, gate, gain, beta):
    value=v['baseline']+gate*(v['gated_history']-v['baseline'])
    value=value+gain*v['occupancy']*(v['current_only']-v['past_background'])
    outputs=np.empty_like(value); outputs[0]=value[0]
    for f in range(1,len(value)):
        keep=beta*(1-v['activity'][f])
        outputs[f]=value[f]*(1-keep)+outputs[f-1]*keep
    return outputs.clip(0,255)

def measure(captures, parameters):
    totals={k:[] for k in ['gap','ghost','static','photo']}; inverted=0; target_count=0
    for c in captures:
        v=c['values']; p=predict(v,**parameters); g=v['GT']; si=c['static']
        # Motion comes from RAW. Training labels affect only the objective.
        for f,(regions,(gi,sign)) in enumerate(zip(c['regions'],c['ghosts'])):
            if c['frames'][f]>=8:
                for ti,ri in regions:
                    true_gap=float(g[f,ti].mean()-np.median(g[f,ri]))
                    gap=float(p[f,ti].mean()-np.median(p[f,ri]))
                    totals['gap'].append(abs(gap-true_gap)); inverted+=int(gap*true_gap<0); target_count+=1
                if len(gi):
                    residual=p[f]-g[f]
                    totals['ghost'].append(float(np.abs(residual[gi]-np.median(residual[si])).mean()))
                totals['photo'].append(float(np.abs(p[f,si]-g[f,si]).mean()))
                if f and c['frames'][f]==c['frames'][f-1]+1:
                    static=si[v['activity'][f,si]<.1]
                    if len(static):
                        d=p[f,static]-p[f-1,static]-(g[f,static]-g[f-1,static])
                        totals['static'].append(float(np.abs(d-np.median(d)).mean()))
    return {k:float(np.mean(v)) if v else None for k,v in totals.items()} | dict(inversions=inverted,targets=target_count)

def run(scene):
    REPORT.mkdir(exist_ok=True)
    captures=collect(scene)
    baseline=measure(captures,dict(gate=0.,gain=0.,beta=0.))
    rows=[]
    for gate,gain,beta in itertools.product([0.,.5,1.],[0.,.25,.5,1.],[0.,.5,.8,.9]):
        parameters=dict(gate=gate,gain=gain,beta=beta)
        metrics=measure(captures,parameters)
        score=sum(weight*metrics[k]/max(baseline[k],.05) for k,weight in [('gap',1.5),('ghost',1.),('static',1.),('photo',.2)] if baseline[k] is not None)
        score+=3*max(0,metrics['inversions']-baseline['inversions'])/max(baseline['targets'],1)
        rows.append(dict(parameters=parameters,metrics=metrics,score=score))
    # Use a candidate that does not increase training inversions or target gap.
    eligible=[r for r in rows if r['metrics']['inversions']<=baseline['inversions'] and
              (baseline['gap'] is None or r['metrics']['gap']<=baseline['gap']+.01)]
    best=min(eligible or rows,key=lambda r:r['score'])
    report=dict(scene=scene,status='train_fit_complete',baseline=baseline,best=best,
                ranked=sorted(rows,key=lambda r:r['score']),fit_split='train',validation_used_for_fit=False,
                test_used=False,weights_frozen=True,train_captures=len(captures),
                tiny_GT_tracks_not_human_identity=True,parameters_frozen_before_validation=True,
                motion_detector_RAW_frames=9,SR_input_frames='original3_or5',speed_optimization=False)
    (REPORT/(scene+'-fit.json')).write_text(json.dumps(report,indent=2))
    print('MOTION14_TRAIN_FIT',json.dumps({k:report[k] for k in ['scene','baseline','best','train_captures']}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True)
    run(p.parse_args().scene)
