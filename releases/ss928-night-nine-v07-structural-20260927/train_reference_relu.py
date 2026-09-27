"""Preserve all reference convolutions and scales; fit ReLU on train contexts."""
import argparse,copy,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-STRUCTURAL-20260927';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from joint_candidates import load_joint
from structural_candidates import replace_gelu
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(64,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(2719);rng=np.random.default_rng(2719);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base=load_joint(a.scene,'baseline',out);teacher=base.core.model.global_reference;teacher.eval()
student=copy.deepcopy(teacher).float();replace_gelu(student)
digest=hashlib.sha256()
for key,value in sorted(teacher.state_dict().items()):digest.update(key.encode());digest.update(value.cpu().numpy().tobytes())
def reference(m,x):
 f=m.encode(x);return m.project(f+f.mean((-2,-1),keepdim=True))
X=[];Y=[];metadata=[]
with torch.no_grad():
 for row in rows:
  xx=[];yy=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();xx.append(ctx.cpu());yy.append(reference(teacher,ctx.half()).float().cpu())
  X.append(torch.cat(xx));Y.append(torch.cat(yy));metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'split':'train'})
X=torch.stack(X);Y=torch.stack(Y);scale=Y.std((0,1,3,4),keepdim=True).clamp_min(.01).reshape(1,16,1,1).cuda()
opt=torch.optim.AdamW(student.parameters(),lr=.0002,weight_decay=0);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter()
for step in range(1,4001):
 ids=rng.integers(0,len(X),4);x=X[ids].reshape(8,1,64,64).cuda();y=Y[ids].reshape(8,16,64,64).cuda();opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):pred=reference(student,x)
 error=(pred.float()-y)/scale;spatial=error.abs().mean()+error.square().mean();ep=error.reshape(4,2,16,64,64);temporal=(ep[:,1]-ep[:,0]).abs().mean();gradient=.5*((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,:,:,1:]-error[:,:,:,:-1]).abs().mean());loss=spatial+temporal+.25*gradient
 scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),1);scaler.step(opt);scaler.update()
 if step%500==0:
  item={'step':step,'loss':float(loss.detach()),'temporal':float(temporal.detach()),'elapsed_s':time.perf_counter()-started};logs.append(item);print('TRAIN',a.scene,item,flush=True)
path=out/f'{a.scene}_reference_relu_004000.pt'
if path.exists():raise FileExistsError(path)
torch.save({'format':'multiscale_reference_relu_v1','scene':a.scene,'step':4000,'reference':{k:v.detach().cpu() for k,v in student.state_dict().items()},'teacher_reference_sha256':digest.hexdigest(),'GT_used':False},path)
(out/f'{a.scene}_reference_relu_training.json').write_text(json.dumps({'scene':a.scene,'seed':2719,'steps':4000,'train_samples':metadata,'teacher_reference_sha256':digest.hexdigest(),'loss':'channel-normalized L1 + MSE + temporal + .25 gradient','GT_used':False,'test_used_for_training_or_checkpoint_selection':False,'NPU_verified':False,'logs':logs},indent=2))
print('COMPLETE',a.scene,flush=True)
