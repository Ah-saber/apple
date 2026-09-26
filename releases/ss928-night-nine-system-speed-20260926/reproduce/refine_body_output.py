"""Distill the half-resolution four-block body into two blocks on training ROIs."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--steps',type=int,default=2000);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from load_packed_model import load_packed_model;from system_variants import CompactBody
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(931);rng=np.random.default_rng(931);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];base=load_packed_model(run/f'{a.scene}_packed_front_{step:06d}.pt',root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt');X=[];Y=[];meta=[];size=96
with torch.no_grad():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[];values=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(cy,cx,ch,cw))[None].cuda();feat=base.core.model.head(base.core.half_input(base.front(stack)));target=base.core.model.body(feat);pair.append(feat);values.append(target)
  for _ in range(6):
   y=int(rng.integers(0,ch//2-size+1));x=int(rng.integers(0,cw//2-size+1));X.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in pair]).cpu());Y.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in values]).cpu());meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'crop_packed':[cy//2+y,cx//2+x,size,size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],len(X),flush=True)
X=torch.stack(X);Y=torch.stack(Y);normalizer=Y.float().std().clamp_min(.01);model=CompactBody(2).cuda().to(memory_format=torch.channels_last)
ck=torch.load(out/f'{a.scene}_body_003000.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['body']);tail_weight=base.core.model.tail[1].rep_conv.weight.detach().float();output_weight=base.core.model.upsample[0].weight.detach().float();del base;torch.cuda.empty_cache();opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);logs=[];start=time.perf_counter();margin=8
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),2);x=X[ids].reshape(4,16,size,size).cuda().float();y=Y[ids].reshape(4,16,size,size).cuda().float();opt.zero_grad(set_to_none=True);raw_error=model(x)-y;display=F.conv2d(F.conv2d(raw_error,tail_weight,padding=1),output_weight,padding=1)[:,:,margin:-margin,margin:-margin]*255.;error=raw_error[:,:,margin:-margin,margin:-margin]/normalizer.to(x.device);spatial=error.abs().mean()*.1+display.abs().mean()+display.square().mean()*.1;ep=display.reshape(2,2,36,size-2*margin,size-2*margin);temporal=(ep[:,1]-ep[:,0]).abs().mean();loss=spatial+temporal;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'temporal':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in [500,1000,2000,3000] or step==a.steps:torch.save({'format':'compact_body_output_v1','scene':a.scene,'step':step,'blocks':2,'body':{k:v.cpu() for k,v in model.state_dict().items()},'GT_used':False},out/f'{a.scene}_body_output_{step:06d}.pt')
(out/f'{a.scene}_body_output_training.json').write_text(json.dumps({'scene':a.scene,'seed':931,'train_samples':meta,'loss':'exact linear tail/output display residual L1 + .1 display MSE + temporal display residual L1 + .1 latent L1','GT_used':False,'logs':logs,'NPU_verified':False},indent=2));print('FINISHED',a.scene,flush=True)
