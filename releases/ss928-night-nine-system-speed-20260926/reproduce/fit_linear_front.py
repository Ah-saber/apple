"""Fit photometrically invariant temporal/spatial linear fronts on train ROIs."""
import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--kernel',type=int,default=2);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from load_packed_model import load_packed_model;from system_variants import LinearFront
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory);rng=np.random.default_rng(932)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];base=load_packed_model(run/f'{a.scene}_packed_front_{step:06d}.pt',root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt');designs=[];targets=[];current=[];meta=[];k=a.kernel;pad=(k-2)//2;phase_indices=[8*k*k+(pad+dy)*k+pad+dx for dy in range(2) for dx in range(2)]
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];ys=rng.integers(24,ch//2-24,256);xs=rng.integers(24,cw//2-24,256);pair_x=[];pair_y=[];pair_c=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(cy,cx,ch,cw))[None].cuda();target=base.front(stack)[0,:,ys,xs].cpu().numpy().T;cur=F.pixel_unshuffle(stack[:,-1:],2)[0,:,ys,xs].cpu().numpy().T
   # Gather only sampled receptive fields; avoid huge full-frame unfold caches.
   chunks=[]
   for dy in range(k):
    for dx in range(k):chunks.append(stack[0,:,torch.as_tensor(ys*2+dy-pad,device='cuda'),torch.as_tensor(xs*2+dx-pad,device='cuda')].cpu().numpy().T)
   features=np.stack(chunks,axis=-1).reshape(256,9*k*k);pair_x.append(features);pair_y.append(target);pair_c.append(cur)
  designs.append(pair_x);targets.append(pair_y);current.append(pair_c);meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'train_roi':row['train_roi_tlhw'],'sampled_packed_yx':np.stack((ys,xs),1).tolist()});print('SAMPLE',a.scene,row['frame_id'],flush=True)
X=np.asarray(designs,dtype=np.float64).reshape(-1,2,256,9*k*k);Y=np.asarray(targets,dtype=np.float64);C=np.asarray(current,dtype=np.float64);model=LinearFront(k)
for phase in range(4):
 features=X-C[:,:,:,phase,None];target=Y[:,:,:,phase]-C[:,:,:,phase];absolute=features.reshape(-1,9*k*k);temporal=(features[:,1]-features[:,0]).reshape(-1,9*k*k);values=target.reshape(-1);delta=(target[:,1]-target[:,0]).reshape(-1);matrix=np.concatenate((absolute,temporal*np.sqrt(.75)),0);rhs=np.concatenate((values,delta*np.sqrt(.75)),0)
 # A bias allows the frozen teacher's small learned phase offsets.
 matrix=np.concatenate((matrix,np.concatenate((np.ones(len(absolute)),np.zeros(len(temporal))))[:,None]),1);ridge=np.eye(matrix.shape[1])*1e-7;ridge[-1,-1]=1e-10;coef=np.linalg.solve(matrix.T@matrix+ridge,matrix.T@rhs);w=coef[:-1];w[phase_indices[phase]]-=w.sum();w[phase_indices[phase]]+=1
 with torch.no_grad():model.conv.weight[phase].copy_(torch.from_numpy(w.reshape(9,k,k).astype(np.float32)));model.conv.bias[phase].fill_(float(coef[-1]))
 print('COEFFICIENT',phase,'sum',float(w.sum()),'current',float(w[phase_indices[phase]]),'bias',float(coef[-1]),flush=True)
torch.save({'format':'linear_front_v1','scene':a.scene,'kernel':k,'front':model.state_dict(),'photometric_weight_sum':model.conv.weight.sum((1,2,3)).tolist(),'GT_used':False},out/f'{a.scene}_linear_k{k}.pt');(out/f'{a.scene}_linear_k{k}_training.json').write_text(json.dumps({'scene':a.scene,'kernel':k,'train_samples':meta,'fit':'least squares teacher packed denoised output + .75 temporal residual difference; affine invariant sum weights=1 per phase','GT_used':False,'NPU_verified':False},indent=2));print('FINISHED',flush=True)
