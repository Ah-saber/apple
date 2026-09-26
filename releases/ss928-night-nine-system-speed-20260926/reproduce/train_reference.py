"""Distill the complete projected thumbnail reference on training contexts."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--steps',type=int,default=2000);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from load_packed_model import load_packed_model;from system_variants import CompactReference
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(64,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(930);rng=np.random.default_rng(930);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];teacher_path=run/f'{a.scene}_packed_front_{step:06d}.pt';base=load_packed_model(teacher_path,root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt');teacher=base.core.model.global_reference;X=[];Y=[];meta=[]
with torch.no_grad():
 for row in rows:
  contexts=[];targets=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();feat=teacher.encode(base.core.half_input(ctx));target=teacher.project(feat+feat.mean((-2,-1),keepdim=True));contexts.append(ctx.cpu());targets.append(target.float().cpu())
  X.append(torch.cat(contexts));Y.append(torch.cat(targets));meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'split':'train'})
X=torch.stack(X);Y=torch.stack(Y);del base,teacher;torch.cuda.empty_cache();model=CompactReference(8).cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);logs=[];start=time.perf_counter();normalizer=Y.std().clamp_min(.01)
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),4);x=X[ids].reshape(8,1,64,64).cuda();y=Y[ids].reshape(8,16,64,64).cuda();opt.zero_grad(set_to_none=True);pred=model(x);error=(pred-y)/normalizer.to(x.device);spatial=error.square().mean()+error.abs().mean();temporal=(error.reshape(4,2,16,64,64)[:,1]-error.reshape(4,2,16,64,64)[:,0]).abs().mean();loss=spatial+temporal;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'temporal':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in [500,1000,2000] or step==a.steps:torch.save({'format':'compact_reference_v1','scene':a.scene,'step':step,'width':8,'reference':{k:v.cpu() for k,v in model.state_dict().items()},'teacher_sha256':hashlib.sha256(teacher_path.read_bytes()).hexdigest(),'GT_used':False},out/f'{a.scene}_reference_{step:06d}.pt')
(out/f'{a.scene}_reference_training.json').write_text(json.dumps({'scene':a.scene,'seed':930,'train_samples':meta,'loss':'normalized MSE + L1 + temporal pair L1','GT_used':False,'logs':logs,'NPU_verified':False},indent=2));print('FINISHED',a.scene,flush=True)
