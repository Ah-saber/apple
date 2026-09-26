"""Train-only activation-weighted output factorization, preserving all 36 phases."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--rank',type=int,default=12);p.add_argument('--steps',type=int,default=3000);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from build_system import load_system;from system_variants import FactorizedOutput
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(1951);rng=np.random.default_rng(1951);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base=load_system(root,a.scene,'base');m=base.core.model;X=[];Y=[];meta=[];size=80;cov=torch.zeros(16,16,dtype=torch.float64);count=0
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pairs=[];values=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));feat=m.body(m.head(base.core.half_input(base.front(stack))));ref=m.global_reference.encode(base.core.half_input(ctx[None].cuda()));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));feat=m.tail(feat+F.interpolate(ref,size=feat.shape[-2:],mode='bilinear',align_corners=False));pairs.append(feat);values.append(m.upsample[0](feat))
  for _ in range(4):
   y=cy//2+int(rng.integers(8,ch//2-size-8));x=cx//2+int(rng.integers(8,cw//2-size-8));xx=torch.cat([v[:,:,y:y+size,x:x+size] for v in pairs]).cpu();X.append(xx);Y.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in values]).cpu());z=xx.double().permute(0,2,3,1).reshape(-1,16);cov+=z.T@z;count+=len(z);meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'train_roi':row['train_roi_tlhw'],'crop_packed':[y,x,size,size]})
  print('CACHE',a.scene,row['frame_id'],flush=True)
X=torch.stack(X);Y=torch.stack(Y);cov/=count;ev,u=torch.linalg.eigh(cov);ev=ev.clamp_min(1e-9);sqrt=(u*ev.sqrt())@u.T;inv=(u*ev.rsqrt())@u.T;old=m.upsample[0];w=old.weight.detach().cpu().double().permute(1,0,2,3).reshape(16,-1);uu,s,vh=torch.linalg.svd(sqrt@w,full_matrices=False);left=inv@uu[:,:a.rank];right=s[:a.rank,None]*vh[:a.rank];scale=left.square().sum(0).sqrt();left/=scale;right*=scale[:,None];model=FactorizedOutput(old,a.rank).float()
with torch.no_grad():model.first.weight.copy_(left.T[:,:,None,None].float().cuda());model.last.weight.copy_(right.reshape(a.rank,36,3,3).permute(1,0,2,3).float().cuda());model.last.bias.copy_(old.bias.float())
del base,m;torch.cuda.empty_cache();opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=0);scaler=torch.amp.GradScaler('cuda');logs=[];start=time.perf_counter();margin=4
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),2);x=X[ids].reshape(4,16,size,size).cuda();y=Y[ids].reshape(4,36,size,size).cuda().float();opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
 error=(pred.float()-y)[:,:,margin:-margin,margin:-margin]*255;ep=error.reshape(2,2,36,size-2*margin,size-2*margin);temporal=(ep[:,1]-ep[:,0]).abs().mean();loss=error.abs().mean()+.05*error.square().mean()+temporal;scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),10);scaler.step(opt);scaler.update()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'spatial_gray':float(error.abs().mean().detach()),'temporal_gray':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,a.rank,item,flush=True)
 if step in [1000,2000,3000] or step==a.steps:torch.save({'format':'weighted_output_v1','rank':a.rank,'output':{k:v.cpu() for k,v in model.state_dict().items()},'GT_used':False,'step':step},out/f'{a.scene}_weighted_r{a.rank}_{step:06d}.pt')
(out/f'{a.scene}_weighted_r{a.rank}_training.json').write_text(json.dumps({'scene':a.scene,'rank':a.rank,'train_samples':meta,'weighted_retained_energy':float(s[:a.rank].square().sum()/s.square().sum()),'cov_eigenvalues':ev.tolist(),'logs':logs,'GT_used':False,'NPU_verified':False},indent=2));print('FINISHED',flush=True)
