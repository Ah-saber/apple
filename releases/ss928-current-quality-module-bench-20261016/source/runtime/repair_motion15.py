"""Train a causal local repair and motion gate on the original training ROIs."""
import json, argparse, random, time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from target_residual import ROOT
from prepare_motion14 import DEST
from fit_motion14 import load_labels

REPORT15=ROOT/'MOTION15-20261013'
FIELDS=['baseline','current_only','gated_history','past_background','activity','occupancy','GT']

def features(v, previous):
    base=v['baseline'];current=v['current_only']
    def local(x):return x-F.avg_pool2d(x,9,1,4,count_include_pad=False)
    return torch.cat([base/255,(current-base)/16,(v['past_background']-base)/16,
                      (previous['baseline']-base)/16,local(base)/16,local(current)/16,
                      v['activity'],v['occupancy'],(v['gated_history']-base)/16,
                      (current-previous['current_only'])/16],1).clamp(-4,4)

class Repair(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(10,24,3,padding=1),nn.ReLU(),
                               nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,3,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():self.net[-1].bias[1]=1.0;self.net[-1].bias[2]=-2.0
    def forward(self,v,previous,previous_output,initial=False):
        heads=self.net(features(v,previous))
        corrected=v['baseline']+16*torch.tanh(heads[:,0:1])
        motion=torch.sigmoid(heads[:,2:3])
        carry=.95*torch.sigmoid(heads[:,1:2])*(1-motion)
        # First causal frame starts with its actual observation.
        output=corrected if initial else corrected*(1-carry)+previous_output*carry
        return output,motion,carry,corrected

def load_captures(scene,raw_targets=False):
    labels=load_labels(scene);captures=[]
    for path in sorted((DEST/scene/'train').glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text());saved=labels[tuple(meta['identity'])]
        if raw_targets:
            saved=torch.load(ROOT/'RAW_TARGET21-20261013'/scene/path.name/'labels.pt',map_location='cpu',weights_only=False)
            if saved['GT_outside_train_ROI_used']:raise ValueError('Original train ROI required')
        if saved['sample']['split']!='train':raise ValueError('Train only')
        arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in FIELDS}
        valid=[i for i,f in enumerate(meta['frames']) if f>=12 and i>=4 and meta['frames'][i]-meta['frames'][i-4]==4]
        tiny=[(i,r) for i in valid for r in saved['regions'].get(meta['frames'][i],[]) if 1<=len(r['target'])<=9]
        macro=[(i,r) for i in valid for r in saved['regions'].get(meta['frames'][i],[]) if r.get('raw_confirmed_foreground') and 1<=len(r['target'])<=1024]
        captures.append(dict(meta=meta,saved=saved,arrays=arrays,valid=valid,tiny=tiny,macro=macro))
    return captures

def patch(c,center,size=192):
    y,x,h,w=c['saved']['roi'];cy,cx=center
    top=int(np.clip(round(cy-size/2),y,y+h-size));left=int(np.clip(round(cx-size/2),x,x+w-size))
    return top,left,size,size

def tensor_at(c,i,crop):
    y,x,h,w=crop
    return {k:torch.from_numpy(np.asarray(a[i,y:y+h,x:x+w],dtype=np.float32).copy())[None,None].cuda()
            for k,a in c['arrays'].items()}

def run(scene,steps=500,gate_only=False,raw_targets=False,raw_gain=False,raw_temporal=False,raw_fixed=False,raw_stable=False):
    if raw_stable:raw_fixed=True
    if raw_fixed:raw_temporal=True
    if raw_temporal:raw_gain=True
    if raw_gain:raw_targets=True
    report_root=ROOT/'MOTION27-20261013' if raw_stable else ROOT/'MOTION25-20261013' if raw_fixed else ROOT/'MOTION24-20261013' if raw_temporal else ROOT/'MOTION23-20261013' if raw_gain else ROOT/'MOTION22-20261013' if raw_targets else ROOT/'MOTION17-20261013' if gate_only else REPORT15
    out=report_root/scene;out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1513);np.random.seed(1513);rng=random.Random(1513)
    captures=load_captures(scene,raw_targets);model=Repair()
    if gate_only or raw_targets:
        if raw_stable:
            from raw_temporal24 import RawFeatures24
            from raw_temporal25 import RawTemporal25
            from raw_stable27 import RawStable27
            parent=torch.load(ROOT/'MOTION25-20261013'/scene/'last_step500.pt',map_location='cpu',weights_only=False)
            model=RawTemporal25();model.load_state_dict(parent['model']);model=RawStable27(model)
        elif raw_temporal:
            from raw_gain23 import RawGain23
            from raw_temporal24 import RawTemporal24,RawFeatures24
            if raw_fixed:
                from raw_temporal25 import RawTemporal25
                RawTemporal24=RawTemporal25
            parent=torch.load(ROOT/'MOTION23-20261013'/scene/'last_step500.pt',map_location='cpu',weights_only=False)
            model=RawGain23();model.load_state_dict(parent['model']);model=RawTemporal24(model)
        elif raw_gain:
            from raw_route22 import RawRoute22
            from raw_gain23 import RawGain23
            parent=torch.load(ROOT/'MOTION22-20261013'/scene/'last_step500.pt',map_location='cpu',weights_only=False)
            model=RawRoute22();model.load_state_dict(parent['model']);model=RawGain23(model)
        else:
            parent=torch.load(REPORT15/scene/'last_step500.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(parent['model'])
        if raw_targets and not raw_gain:
            from raw_route22 import RawRoute22
            model=RawRoute22(model)
        elif not raw_gain:
            from repair_gate17 import Repair17
            model=Repair17(model)
    model=model.cuda();optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.0007)
    raw_provider=RawFeatures24(scene) if raw_temporal else None
    history=[];started=time.time();target_visits=0
    for step in range(1,steps+1):
        c=captures[(step-1)%len(captures)]
        if raw_targets and c['macro'] and step%4 in [1,2]:
            i,region=rng.choice(c['macro']);center=np.asarray(region['target']).mean(0)
        elif c['tiny'] and step%3:
            i,region=rng.choice(c['tiny']);center=np.asarray(region['target']).mean(0)
        else:
            i=rng.choice(c['valid']);y,x,h,w=c['saved']['roi'];center=(rng.randrange(y+96,y+h-95),rng.randrange(x+96,x+w-95))
        crop=patch(c,center);top,left,h,w=crop
        sequence=[tensor_at(c,j,crop) for j in range(i-4,i+1)]
        if raw_temporal:
            for j,v in zip(range(i-4,i+1),sequence):v['raw24']=raw_provider.for_record(c['meta']['row'],c['meta']['frames'][j],crop)
        previous=sequence[0];previous_output=previous['baseline'];loss=0.;notes={};ntiny=0
        for offset,v in enumerate(sequence[1:],start=1):
            pred,motion,carry,corrected=model(v,previous,previous_output)
            gt=v['GT'];gt_prev=previous['GT'];f=c['meta']['frames'][i-4+offset]
            dg=gt-gt_prev;dg=dg-dg.median()
            smooth=F.avg_pool2d(dg,3,1,1,count_include_pad=False)
            moving=F.max_pool2d((smooth.abs()>2.5).float(),5,1,2)
            regions=c['saved']['regions'].get(f,[]);gaps=[];margins=[];focus_gates=[]
            for r in regions:
                t=np.asarray(r['target'],int)-[top,left];ring=np.asarray(r['ring'],int)-[top,left]
                if not 1<=len(t)<=(1024 if raw_targets else 9) or len(ring)<3:continue
                if not (np.all(t>=8) and np.all(ring>=4) and np.all(t[:,0]<h-8) and np.all(t[:,1]<w-8) and np.all(ring[:,0]<h-4) and np.all(ring[:,1]<w-4)):continue
                ti=torch.tensor(t[:,0]*w+t[:,1],device='cuda');ri=torch.tensor(ring[:,0]*w+ring[:,1],device='cuda')
                pflat=pred.flatten();gflat=gt.flatten()
                gap=pflat[ti].mean()-pflat[ri].median();true=gflat[ti].mean()-gflat[ri].median()
                gaps.append((gap-true).abs());margins.append(F.relu(.9*true.abs()-true.sign()*gap))
                moving.flatten()[ti]=1;ntiny+=1
                if gate_only or raw_targets:focus_gates.append(carry.flatten()[ti].mean()-torch.log(motion.flatten()[ti].clamp_min(1e-5)).mean())
            photo=F.smooth_l1_loss(pred[...,8:-8,8:-8],gt[...,8:-8,8:-8],beta=2.)
            pred_local=pred-F.avg_pool2d(pred,7,1,3,count_include_pad=False)
            gt_local=gt-F.avg_pool2d(gt,7,1,3,count_include_pad=False)
            detail=(pred_local-gt_local).abs()[...,8:-8,8:-8].mean()
            static=1-F.max_pool2d(moving,7,1,3)
            temporal=(pred-previous_output) if raw_stable else (pred-previous_output)-(gt-gt_prev)
            temporal=temporal-temporal[...,8:-8,8:-8].median()
            temporal_loss=(temporal.abs()*static)[...,8:-8,8:-8].mean()
            ghost=np.asarray(c['saved']['labels'].get(f,{}).get('ghost',np.empty((0,3))),int).reshape(-1,3)
            ghost_loss=pred.sum()*0
            if len(ghost):
                gy=ghost[:,0]-top;gx=ghost[:,1]-left;valid=(gy>=8)&(gy<h-8)&(gx>=8)&(gx<w-8)
                if valid.any():
                    gi=torch.tensor(gy[valid]*w+gx[valid],device='cuda')
                    residual=pred-gt;centered=residual-F.avg_pool2d(residual,21,1,10,count_include_pad=False)
                    ghost_loss=centered.flatten()[gi].abs().mean();moving.flatten()[gi]=1
                    if gate_only or raw_targets:focus_gates.append(.5*(carry.flatten()[gi].mean()-torch.log(motion.flatten()[gi].clamp_min(1e-5)).mean()))
            motion_loss=F.binary_cross_entropy(motion[...,8:-8,8:-8],moving[...,8:-8,8:-8],reduction='none')
            motion_loss=(motion_loss*(1+5*moving[...,8:-8,8:-8])).mean()
            gap_loss=torch.stack(gaps).mean() if gaps else pred.sum()*0
            margin=torch.stack(margins).mean() if margins else pred.sum()*0
            if gate_only or raw_targets:
                focus=torch.stack(focus_gates).mean() if focus_gates else pred.sum()*0
                current_loss=photo+.5*detail+3*temporal_loss+8*gap_loss+4*margin+3*ghost_loss+.4*motion_loss+4*focus
            else:
                current_loss=photo+.5*detail+2*temporal_loss+4*gap_loss+3*margin+2*ghost_loss+.4*motion_loss
            loss=loss+current_loss/4
            previous,previous_output=v,pred
            notes=dict(photo=float(photo.detach()),detail=float(detail.detach()),static=float(temporal_loss.detach()),
                       gap=float(gap_loss.detach()),ghost=float(ghost_loss.detach()),carry=float(carry.mean().detach()),motion=float(motion.mean().detach()))
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        for group in optimizer.param_groups:group['lr']=.0007*(1-.8*step/steps)
        target_visits+=ntiny
        if step==1 or step%25==0:
            record=dict(step=step,total=float(loss.detach()),tiny_in_unroll=ntiny,**notes);history.append(record)
            (out/'progress.json').write_text(json.dumps(record));print('MOTION15_STEP',scene,json.dumps(record),flush=True)
    checkpoint=dict(method='RAW_STABLE27' if raw_stable else 'RAW_TEMPORAL25' if raw_fixed else 'RAW_TEMPORAL24' if raw_temporal else 'RAW_GAIN23' if raw_gain else 'RAW_ROUTE22' if raw_targets else 'TARGET_GATE17' if gate_only else 'CAUSAL_REPAIR15',scene=scene,steps=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_rng=rng.getstate(),
                    numpy_rng=np.random.get_state(),training_ROIs=[c['saved']['roi'] for c in captures],gate_only=gate_only,raw_targets=raw_targets,raw_gain=raw_gain,raw_temporal=raw_temporal,
                    input_count='Original 3/5 SR + nine-frame RAW detector',fit_split='train',validation_used=False,test_used=False)
    torch.save(checkpoint,out/'last_step500.pt')
    summary=dict(scene=scene,steps=steps,train_captures=len(captures),tiny_visits=target_visits,history=history,
                 elapsed_s=time.time()-started,SR_body_frozen=True,causal_unroll=4,validation_used=False,test_used=False,
                 original_raw_path=True,speed_optimization=False,weights=str(out/'last_step500.pt'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('MOTION15_TRAIN_DONE',json.dumps(summary),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gate-only',action='store_true');p.add_argument('--raw-targets',action='store_true');p.add_argument('--raw-gain',action='store_true');p.add_argument('--raw-temporal',action='store_true');p.add_argument('--raw-fixed',action='store_true');p.add_argument('--raw-stable',action='store_true');a=p.parse_args();run(a.scene,gate_only=a.gate_only,raw_targets=a.raw_targets,raw_gain=a.raw_gain,raw_temporal=a.raw_temporal,raw_fixed=a.raw_fixed,raw_stable=a.raw_stable)
