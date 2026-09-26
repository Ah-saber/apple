"""Train a direct-quarter front against frozen v0.6, using training ROIs only."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--steps',type=int,default=3000);p.add_argument('--width',type=int);p.add_argument('--blocks',type=int,default=2);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';out.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from load_packed_model import load_packed_model
from system_variants import ContrastStatsFront
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(929);rng=np.random.default_rng(929);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];teacher_path=run/f'{a.scene}_packed_front_{step:06d}.pt';cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt';teacher=load_packed_model(teacher_path,cp,bp).front
for parameter in teacher.parameters():parameter.requires_grad_(False)
inputs=[];targets=[];meta=[];size=192
with torch.no_grad():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[];values=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(cy,cx,ch,cw))[None].cuda();pair.append(x);values.append(teacher(x))
  for _ in range(6):
   y=int(rng.integers(0,(ch-size)//4+1))*4;x=int(rng.integers(0,(cw-size)//4+1))*4;inputs.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in pair]).cpu());targets.append(torch.cat([v[:,:,y//2:(y+size)//2,x//2:(x+size)//2] for v in values]).cpu());meta.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'crop':[cy+y,cx+x,size,size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],len(inputs),flush=True)
  del pair,values
X=torch.stack(inputs);Y=torch.stack(targets);del inputs,targets,teacher;torch.cuda.empty_cache();front=ContrastStatsFront(a.width or (8 if not special else 12)).cuda().to(memory_format=torch.channels_last);ck=torch.load(out/f'{a.scene}_balanced_w{front.width}_003000.pt',map_location='cpu',weights_only=True);front.load_state_dict({k:v for k,v in ck['front'].items() if k!='stats_weight'},strict=False);scaler=torch.amp.GradScaler('cuda');opt=torch.optim.AdamW(front.parameters(),lr=.0001,weight_decay=.0001);logs=[];start=time.perf_counter();margin=24
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),2);x=X[ids].reshape(4,9,size,size).cuda();target=Y[ids].reshape(4,4,size//2,size//2).cuda();opt.zero_grad(set_to_none=True);
 with torch.autocast('cuda',dtype=torch.float16):pred=front(x)
 error=(pred-target)[:,:,margin:-margin,margin:-margin]/.025
 absolute=error.abs().mean();squared=error.square().mean();gradient=.5*((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,:,:,1:]-error[:,:,:,:-1]).abs().mean());ep=error.reshape(2,2,4,size//2-2*margin,size//2-2*margin);temporal=(ep[:,1]-ep[:,0]).abs().mean();loss=absolute+squared+.25*gradient+.75*temporal;scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(front.parameters(),1.);scaler.step(opt);scaler.update()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'absolute':float(absolute.detach()),'temporal':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in [500,1000,2000,3000] or step==a.steps:torch.save({'format':'balanced_centered_front_v1','scene':a.scene,'step':step,'width':front.width,'blocks':front.blocks,'front':{k:v.cpu() for k,v in front.state_dict().items()},'teacher_sha256':hashlib.sha256(teacher_path.read_bytes()).hexdigest(),'GT_used':False},out/f'{a.scene}_contrast_w{front.width}_{step:06d}.pt')
(out/f'{a.scene}_contrast_w{front.width}_training.json').write_text(json.dumps({'scene':a.scene,'steps':a.steps,'seed':929,'GT_used':False,'teacher':str(teacher_path),'teacher_sha256':hashlib.sha256(teacher_path.read_bytes()).hexdigest(),'train_samples':meta,'loss':'L1 + squared + .25 error spatial gradient + .75 pair temporal error','margin_packed_pixels':margin,'logs':logs,'NPU_verified':False},indent=2));print('FINISHED',a.scene,flush=True)
