"""Train a 16-channel 3-convolution reference against current frozen teacher.
Train contexts and spatial ROI only; fixed 16000 steps; no GT optimization.
"""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-REFERENCE16-20260927';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from reference_candidates import load_reference,Reference16,BASE_CASE,prepare_inputs
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2785);rng=np.random.default_rng(2785);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
teacher=load_reference(a.scene,BASE_CASE,out);teacher.requires_grad_(False);m=teacher.core.model;X=[];Y=[];B=[];grids=[];metadata=[];size=64
with torch.no_grad():
 for row in rows:
  context=[];targets=[];features=[];cy,cx,ch,cw=row['train_roi_tlhw'];assert cy%2==cx%2==0
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);features.append(m.body(m.head(teacher.core.half_input(teacher.front(x)))))
   ref=m.global_reference.encode(teacher.core.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));context.append(ctx.cpu());targets.append(ref.cpu())
  for _ in range(4):
   y=cy//2+int(rng.integers(0,ch//2-size+1));x=cx//2+int(rng.integers(0,cw//2-size+1));X.append(torch.cat(context));Y.append(torch.cat(targets));B.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in features]).cpu());gy,gx=torch.meshgrid(torch.arange(y,y+size,dtype=torch.float32),torch.arange(x,x+size,dtype=torch.float32),indexing='ij');grids.append(torch.stack((2*(gx+.5)/640-1,2*(gy+.5)/512-1),-1));metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[y*2,x*2,size*2,size*2],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
X=torch.stack(X);Y=torch.stack(Y);B=torch.stack(B);grids=torch.stack(grids);train_count=len(metadata)-16;assert train_count>0
reference=copy.deepcopy(m.global_reference).float().requires_grad_(False);tail=copy.deepcopy(m.tail).float().requires_grad_(False);projection=copy.deepcopy(teacher.output.conv).float().requires_grad_(False);del teacher,features,m,ref,ctx;torch.cuda.empty_cache()
student=Reference16().cuda().to(memory_format=torch.channels_last)
with torch.no_grad():student.last.weight.zero_();student.last.bias.copy_(Y[:train_count].float().mean((0,1,3,4)).cuda()/2)
initial={k:v.detach().cpu().clone() for k,v in student.state_dict().items()};scale=Y[:train_count].float().std((0,1,3,4)).clamp_min(.01).cuda().reshape(1,16,1,1);opt=torch.optim.AdamW(student.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,16000,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda');started=time.perf_counter();logs=[]
def loss_for(ids,training):
 context=X[ids].reshape(-1,1,64,64).cuda();body=B[ids].reshape(-1,16,size,size).float().cuda();grid=grids[ids].repeat_interleave(2,0).cuda()
 if training:
  apply=(torch.rand(len(ids),1,1,1,device='cuda')<.5).repeat_interleave(2,0);gain=(.95+.1*torch.rand(len(ids),1,1,1,device='cuda')).repeat_interleave(2,0);offset=((torch.rand(len(ids),1,1,1,device='cuda')-.5)*.02).repeat_interleave(2,0);context=torch.where(apply,context*gain+offset,context)
 with torch.autocast('cuda',dtype=torch.float16):
  with torch.no_grad():v=reference.encode(context.half());target=reference.project(v+v.mean((-2,-1),keepdim=True))
  pred=student(context)
  # Sampling only training crops is equivalent to full align_corners=False
  # interpolation at the same coordinates in exact arithmetic.
  pred_crop=F.grid_sample(pred.float(),grid,padding_mode='border',align_corners=False);target_crop=F.grid_sample(target.float(),grid,padding_mode='border',align_corners=False)
  phase=projection(tail(body+pred_crop));target_phase=projection(tail(body+target_crop))
 error=(pred.float()-target.float())/scale;feature=error.abs().mean()+.5*error.square().mean();fp=error.reshape(-1,2,16,64,64);feature_temporal=(fp[:,1]-fp[:,0]).abs().mean();gray=(phase[:,:9].float().clamp(0,1)-target_phase[:,:9].float().clamp(0,1))[:,:,2:-2,2:-2]*255;pair=gray.reshape(-1,2,9,size-4,size-4);temporal=(pair[:,1]-pair[:,0]).abs().mean();local=F.avg_pool2d(gray,8,8).abs().mean();loss=2*feature+2*feature_temporal+gray.abs().mean()+.1*gray.square().mean()+2*temporal+2*local
 return loss,{'reference_normalized_mae':float(error.abs().mean().detach()),'phase_mae_gray':float(gray.abs().mean().detach()),'temporal_mae_gray':float(temporal.detach()),'local_mean_mae_gray':float(local.detach())}
for step in range(1,16001):
 ids=rng.integers(0,train_count,2);opt.zero_grad(set_to_none=True);loss,info=loss_for(ids,True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),1);scaler.step(opt);scaler.update();scheduler.step()
 if step%1000==0:
  with torch.no_grad():_,held=loss_for(np.arange(train_count,train_count+2),False)
  item={'step':step,'loss':float(loss.detach()),**info,'train_roi_holdout_first_two_pairs':held,'elapsed_s':time.perf_counter()-started};logs.append(item);print('TRAIN',a.scene,item,flush=True)
path=out/f'{a.scene}_reference16_016000.pt'
if path.exists():raise FileExistsError(path)
torch.save({'format':'reference16_teacher_output_v1','scene':a.scene,'step':16000,'reference':{k:v.detach().cpu() for k,v in student.state_dict().items()},'initial_reference':initial,'teacher_case':BASE_CASE,'GT_used':False},path)
(out/f'{a.scene}_reference16_training.json').write_text(json.dumps({'scene':a.scene,'steps':16000,'seed':2785,'train_samples':metadata[:train_count],'train_roi_holdout_samples':metadata[train_count:],'teacher_case':BASE_CASE,'GT_used':False,'test_used_for_training_or_checkpoint_selection':False,'augmentation':'50% pairs: same context gain .95..1.05 and offset -.01.. .01; teacher queried on augmented context','loss':'2 normalized reference L1/.5MSE + 2 reference temporal + output phase L1/.1MSE + 2 output temporal + 2 local mean','source_geometry':'grid_sample of 64x64 projected reference at full 512x640 feature coordinates; only train ROI crop gradients','logs':logs,'NPU_verified':False},indent=2));print('COMPLETE',a.scene,flush=True)
