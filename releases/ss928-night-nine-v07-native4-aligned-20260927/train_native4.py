"""Train four native-pixel outputs against rounded/clipped frozen teacher outputs."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from continuation_candidates import load_continuation,NativeFourOutput
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2783);rng=np.random.default_rng(2783);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
teacher=load_continuation(a.scene,'baseline',out);original=teacher.core.model.upsample[0];students={k:NativeFourOutput(original,k).float().to(memory_format=torch.channels_last) for k in ['linear','relu8']};X=[];Y=[];metadata=[];size=64
with torch.no_grad():
 for row in rows:
  features=[];targets=[];cy,cx,ch,cw=row['train_roi_tlhw'];assert cy%2==0 and cx%2==0
  for frame in [max(0,row['frame_id']-1),row['frame_id']]:
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();c,m=teacher.core,teacher.core.model
   v=m.body(m.head(c.half_input(teacher.front(x))));ref=m.global_reference.encode(c.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));v=m.tail(v+F.interpolate(ref,size=v.shape[-2:],mode='bilinear',align_corners=False))
   phases=original(v).float().clamp(0,1).mul(255).round();native=phases.reshape(1,2,3,2,3,*v.shape[-2:]).mean((2,4)).reshape(1,4,*v.shape[-2:])/255;features.append(v);targets.append(native)
  for _ in range(6):
   y=cy//2+int(rng.integers(0,ch//2-size+1));x=cx//2+int(rng.integers(0,cw//2-size+1));X.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in features]).cpu());Y.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in targets]).cpu());metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[y*2,x*2,size*2,size*2],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
X=torch.stack(X);Y=torch.stack(Y);del teacher;torch.cuda.empty_cache()
for kind,student in students.items():
 opt=torch.optim.AdamW(student.parameters(),lr=1e-5 if kind=='linear' else 2e-5,weight_decay=0);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter();local_rng=np.random.default_rng(2783)
 for step in range(1,4001):
  ids=local_rng.integers(0,len(X),2);x=X[ids].reshape(4,16,size,size).cuda();target=Y[ids].reshape(4,4,size,size).cuda()[:,:,1:-1,1:-1];opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):pred=student.phases(x)[:,:,1:-1,1:-1]
  error=(pred.float().clamp(0,1)-target)*255;paired=error.reshape(2,2,4,size-2,size-2);temporal_error=paired[:,1]-paired[:,0];dt=(target.reshape(2,2,4,size-2,size-2)[:,1]-target.reshape(2,2,4,size-2,size-2)[:,0])*255;mask=(dt.abs()>=.5)&(dt.abs()<=8);weak=temporal_error[mask].abs().mean() if mask.any() else temporal_error.abs().mean();temporal=temporal_error.abs().mean();local_mean=F.avg_pool2d(error,8,8).abs().mean();gradient=.5*((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,:,:,1:]-error[:,:,:,:-1]).abs().mean());loss=error.abs().mean()+.5*error.square().mean()+2*temporal+2*weak+2*local_mean+.5*gradient
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),1);scaler.step(opt);scaler.update()
  if step%500==0:
   item={'step':step,'loss':float(loss.detach()),'native_mae_gray':float(error.abs().mean().detach()),'temporal_mae_gray':float(temporal.detach()),'weak_temporal_mae_gray':float(weak.detach()),'local_mean_error_gray':float(local_mean.detach()),'elapsed_s':time.perf_counter()-started};logs.append(item);print('TRAIN',a.scene,kind,item,flush=True)
 path=out/f'{a.scene}_native4_{kind}_004000.pt'
 if path.exists():raise FileExistsError(path)
 torch.save({'format':'native_four_teacher_clipped_v1','scene':a.scene,'kind':kind,'step':4000,'output':{k:v.detach().cpu() for k,v in student.state_dict().items()},'GT_used':False},path)
 (out/f'{a.scene}_native4_{kind}_training.json').write_text(json.dumps({'scene':a.scene,'seed':2783,'steps':4000,'train_samples':metadata,'GT_used':False,'test_used_for_training_or_checkpoint_selection':False,'teacher':'frozen v0.7 front_f32; full-feature geometry; clipped rounded native phase means','loss':'native L1 + .5 MSE + 2 temporal + 2 teacher weak temporal + 2 local mean + .5 gradient','logs':logs,'NPU_verified':False},indent=2))
 print('COMPLETE',a.scene,kind,flush=True)
