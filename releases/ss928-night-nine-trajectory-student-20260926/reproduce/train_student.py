"""Pilot feature distillation using original train ROIs only, never held-out GT."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));p.add_argument('--steps',type=int,default=2000);p.add_argument('--compact',action='store_true');a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs'/('SS928-TRAJECTORY-STUDENT-COMPACT-20260926' if a.compact else 'SS928-TRAJECTORY-STUDENT-20260926');out.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime' if (Path(__file__).parent/'runtime').is_dir() else Path(__file__).parent.parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from trajectory_student import TrajectoryStudent
from load_candidate import load_candidate
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
ds=dataset_for_config(state['config'],'train');rows=[r for r in ds.records if r['scene_id']=='night_'+a.scene]
rows=[rows[int(i)] for i in np.linspace(0,len(rows)-1,32,dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(928);rng=np.random.default_rng(928)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
candidate=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt';baseline=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt'
teacher=load_candidate(candidate,baseline,remove_zero_init=True)
for param in teacher.parameters():param.requires_grad_(False)
inputs=[];targets=[];meta=[];size=192
with torch.no_grad():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];stack=ds.normalized_stack(row,(cy,cx,ch,cw))[None].cuda()
  features=teacher.model._trajectory_features(stack)
  for _ in range(8):
   y=int(rng.integers(0,(ch-size)//4+1))*4;x=int(rng.integers(0,(cw-size)//4+1))*4
   inputs.append(stack[0,:,y:y+size,x:x+size].cpu().float() if a.compact else stack[0,:,y:y+size,x:x+size].cpu().half());targets.append(features[0,:,y:y+size,x:x+size].cpu().half())
   meta.append({'sample':row['sample_id'],'frame':row['frame_id'],'crop':[cy+y,cx+x,size,size],'train_roi':row['train_roi_tlhw']})
  del stack,features
  print('CACHE',a.scene,row['frame_id'],len(inputs),flush=True)
X=torch.stack(inputs);Y=torch.stack(targets);del inputs,targets;torch.cuda.empty_cache()
# Only student parameters update. Validate against an independent split after training.
student=TrajectoryStudent(width=8 if a.compact else 12,quarter_stem=a.compact).cuda().to(memory_format=torch.channels_last)
opt=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=.0001)
logs=[];start=time.perf_counter();margin=48;sl=(slice(None),slice(None),slice(margin,-margin),slice(margin,-margin))
limits=torch.tensor([10.,20.,10.,20.],device='cuda').reshape(1,4,1,1)
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),4);x=X[ids].cuda().float();target=Y[ids].cuda().float()
 opt.zero_grad(set_to_none=True)
 pred=student(x);pn=pred[sl]/limits;tn=target[sl]/limits
 loss=((pn-tn).abs()*(1+4*tn)).mean()
 loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step()
 if step%100==0:
  item={'step':step,'weighted_normalized_feature_l1':float(loss.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step%500==0 or step==a.steps:
  torch.save({'format':'trajectory_student_v1','scene':a.scene,'step':step,'width':8 if a.compact else 12,'quarter_stem':a.compact,'student':{k:v.cpu() for k,v in student.state_dict().items()},'teacher_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'training_gt_used':False},out/f'{a.scene}_student_{step:06d}.pt')
(out/f'{a.scene}_training.json').write_text(json.dumps({'scene':a.scene,'steps':a.steps,'seed':928,'source_teacher':str(candidate),'teacher_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'train_samples':meta,'GT_used':False,'cache_input_dtype':'float32' if a.compact else 'float16','cache_target_dtype':'float16','loss_margin':margin,'logs':logs,'parameters':sum(v.numel() for v in student.parameters()),'quarter_stem':a.compact,'architecture':('FP32 temporal mean/subtraction; stride4 Conv5 9->8; four residual Conv5 8->8 at quarter resolution; ConvTranspose8 stride4 8->4; sigmoid limits10/20/10/20' if a.compact else 'FP32 temporal mean/subtraction; stride2 Conv5 9->8; stride2 Conv3 8->12; four residual Conv5 12->12 at quarter resolution; ConvTranspose8 stride4 12->4; sigmoid limits10/20/10/20'),'npu_compatibility_verified':False},indent=2))
print('FINISHED',a.scene,flush=True)
