"""Refine front with frozen complete mapping and strong temporal output supervision."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--steps',type=int,default=3000);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from build_system import load_system
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(391);rng=np.random.default_rng(391);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
teacher=load_system(root,a.scene,'base');student=load_system(root,a.scene,'balanced');front=student.front.float();m=student.core.model
for v in m.parameters():v.requires_grad_(False)
X=[];Y=[];R=[];P=[];meta=[];size=192
with torch.no_grad():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pairs=[];values=[];refs=[];packed=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));pf=teacher.front(stack);feat=m.body(m.head(student.core.half_input(pf)));ref=m.global_reference.encode(student.core.half_input(ctx[None].cuda()));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));ref=F.interpolate(ref,size=feat.shape[-2:],mode='bilinear',align_corners=False);pairs.append(stack);packed.append(pf);refs.append(ref);values.append(m.upsample[0](m.tail(feat+ref)))
  for _ in range(4):
   y=(cy+int(rng.integers(16,ch-size-16)))//2*2;x=(cx+int(rng.integers(16,cw-size-16)))//2*2;X.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in pairs]).half().cpu());Y.append(torch.cat([v[:,:,y//2:y//2+size//2,x//2:x//2+size//2] for v in values]).cpu());R.append(torch.cat([v[:,:,y//2:y//2+size//2,x//2:x//2+size//2] for v in refs]).cpu());P.append(torch.cat([v[:,:,y//2:y//2+size//2,x//2:x//2+size//2] for v in packed]).half().cpu());meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'train_roi':row['train_roi_tlhw'],'crop_raw':[y,x,size,size]})
  print('CACHE',a.scene,row['frame_id'],flush=True)
X=torch.stack(X);Y=torch.stack(Y);R=torch.stack(R);P=torch.stack(P);del teacher;torch.cuda.empty_cache();opt=torch.optim.AdamW(front.parameters(),lr=.00004,weight_decay=0);scaler=torch.amp.GradScaler('cuda');logs=[];start=time.perf_counter();margin=16
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),2);x=X[ids].reshape(4,9,size,size).cuda();y=Y[ids].reshape(4,36,size//2,size//2).cuda().float();r=R[ids].reshape(4,16,size//2,size//2).cuda();pp=P[ids].reshape(4,4,size//2,size//2).cuda().float();opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):
  fp=front(x);feat=m.body(m.head(student.core.half_input(fp)));pred=m.upsample[0](m.tail(feat+r))
 error=(pred.float().clamp(0,1)-y.clamp(0,1))[:,:,margin:-margin,margin:-margin]*255;ep=error.reshape(2,2,36,size//2-2*margin,size//2-2*margin);temporal=(ep[:,1]-ep[:,0]).abs().mean();ferr=(fp.float()-pp)[:,:,margin:-margin,margin:-margin]*255;loss=error.abs().mean()+.05*error.square().mean()+3*temporal+.25*ferr.abs().mean();scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(front.parameters(),10);scaler.step(opt);scaler.update()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'spatial_gray':float(error.abs().mean().detach()),'temporal_gray':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in [1000,2000,3000] or step==a.steps:torch.save({'format':'balanced_output_front_v1','width':front.width,'front':{k:v.cpu() for k,v in front.state_dict().items()},'GT_used':False,'step':step},out/f'{a.scene}_front_output_{step:06d}.pt')
(out/f'{a.scene}_front_output_training.json').write_text(json.dumps({'scene':a.scene,'train_samples':meta,'logs':logs,'GT_used':False,'NPU_verified':False,'temporal_weight':3},indent=2));print('FINISHED',flush=True)
