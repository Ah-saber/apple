"""Distill original denoised packed2 tensor from train ROIs with temporal-pair loss."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));p.add_argument('--steps',type=int,default=10000);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-COMPACT8-20260926';out.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from packed_front import PackedFront
from load_candidate import load_candidate
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
ds=dataset_for_config(state['config'],'train');rows=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[rows[int(i)] for i in np.linspace(0,len(rows)-1,32,dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(928);rng=np.random.default_rng(928)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt'
teacher=load_candidate(cp,bp,remove_zero_init=True)
for parameter in teacher.parameters():parameter.requires_grad_(False)
inputs=[];targets=[];weights=[];metadata=[];size=192
with torch.no_grad():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[];denoised=[];maps=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(cy,cx,ch,cw))[None].cuda();features=teacher.model._trajectory_features(stack)
   joined=teacher.second(F.relu(teacher.first(teacher.half_input(torch.cat((stack,features),1))))).float()
   value=stack[:,-1:].float()+joined[:,:1]*(1.-torch.sigmoid(joined[:,1:]))
   pair.append(stack);denoised.append(value);maps.append(1.+(features[:,1:2]+features[:,3:4]).clamp(0,20)*.1)
  for _ in range(8):
   y=int(rng.integers(0,(ch-size)//4+1))*4;x=int(rng.integers(0,(cw-size)//4+1))*4
   inputs.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in pair]).cpu())
   targets.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in denoised]).cpu())
   weights.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in maps]).cpu().half())
   metadata.append({'sample':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'crop':[cy+y,cx+x,size,size],'train_roi':row['train_roi_tlhw']})
  del pair,denoised,maps,stack,features,joined,value
  print('CACHE',a.scene,row['frame_id'],len(inputs),flush=True)
X=torch.stack(inputs);Y=torch.stack(targets);W=torch.stack(weights);del inputs,targets,weights,teacher;torch.cuda.empty_cache()
front=PackedFront(width=8).cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(front.parameters(),lr=.001,weight_decay=.0001)
margin=48;sl=(slice(None),slice(None),slice(margin,-margin),slice(margin,-margin));logs=[];start=time.perf_counter()
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),2);x=X[ids].reshape(4,9,size,size).cuda();target=Y[ids].reshape(4,1,size,size).cuda();w=W[ids].reshape(4,1,size,size).cuda().float()
 opt.zero_grad(set_to_none=True);packed=front(x);pred=F.pixel_shuffle(packed,2)
 error=(pred[sl]-target[sl])/.025;weight=w[sl]
 absolute=(error.abs()*weight).mean();squared=(error.square()*weight).mean()
 gradient=((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,:,:,1:]-error[:,:,:,:-1]).abs().mean())*.5
 ep=error.reshape(2,2,1,size-2*margin,size-2*margin);temporal=(ep[:,1]-ep[:,0]).abs().mean()
 loss=absolute+squared+.25*gradient+.5*temporal
 loss.backward();torch.nn.utils.clip_grad_norm_(front.parameters(),1.);opt.step()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'absolute':float(absolute.detach()),'temporal':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in (500,1000,2000,5000,10000) or step==a.steps:
  torch.save({'format':'packed_front_v1','scene':a.scene,'step':step,'width':8,'front':{k:v.cpu() for k,v in front.state_dict().items()},'teacher_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'GT_used':False},out/f'{a.scene}_packed_front_{step:06d}.pt')
(out/f'{a.scene}_training.json').write_text(json.dumps({'scene':a.scene,'steps':a.steps,'seed':928,'teacher':str(cp),'teacher_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'train_samples':metadata,'GT_used':False,'input_target_cache_dtype':'float32','loss_margin':margin,'loss':'weighted L1 + weighted MSE + .25 spatial error gradient + .5 temporal pair error','logs':logs,'parameters':sum(v.numel() for v in front.parameters()),'architecture':'10 input channels: nine FP32 residuals*64 plus current; Conv6 stride2 10->8; Conv3 stride2 8->8; four quarter-resolution Conv5 8->8; ConvTranspose4 stride2 8->4; half-resolution Conv1 8->4 skip; packed current + correction*.025','NPU_verified':False},indent=2))
print('FINISHED',a.scene,flush=True)
