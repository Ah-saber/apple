"""Refine the warm-start approximate reference using feature and display sensitivity."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--steps',type=int,default=2000);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from load_packed_model import load_packed_model;from system_variants import SiLUReference
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(934);rng=np.random.default_rng(934);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];base=load_packed_model(run/f'{a.scene}_packed_front_{step:06d}.pt',root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt');old=base.core.model.global_reference;model=SiLUReference(old).cuda().float().to(memory_format=torch.channels_last);X=[];Y=[];meta=[]
with torch.no_grad():
 for row in rows:
  pair=[];target=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();feat=old.encode(base.core.half_input(ctx));value=old.project(feat+feat.mean((-2,-1),keepdim=True));pair.append(ctx.cpu());target.append(value.cpu())
  X.append(torch.cat(pair));Y.append(torch.cat(target));meta.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'split':'train'})
X=torch.stack(X);Y=torch.stack(Y);dc=base.core.model.upsample[0].weight.float().sum((-2,-1))@base.core.model.tail[1].rep_conv.weight.float().sum((-2,-1));dc=dc[:,:,None,None].detach()*255.;normalizer=Y.float().std().clamp_min(.01);del old,base;torch.cuda.empty_cache();opt=torch.optim.AdamW(model.parameters(),lr=.00001,weight_decay=0);scaler=torch.amp.GradScaler('cuda');logs=[];start=time.perf_counter()
for step in range(1,a.steps+1):
 ids=rng.integers(0,len(X),4);x=X[ids].reshape(8,1,64,64).cuda();y=Y[ids].reshape(8,16,64,64).cuda().float();opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
 err=pred.float()-y;feature=err.abs().mean()/normalizer.to(x.device);display=F.conv2d(err,dc);ep=display.reshape(4,2,36,64,64);temporal=(ep[:,1]-ep[:,0]).abs().mean();loss=feature+.5*display.abs().mean()+temporal;scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
 if step%200==0:
  item={'step':step,'loss':float(loss.detach()),'display_mae_gray_proxy':float(display.detach().abs().mean()),'temporal_gray_proxy':float(temporal.detach()),'elapsed_s':time.perf_counter()-start};logs.append(item);print('TRAIN',a.scene,item,flush=True)
 if step in [500,1000,2000] or step==a.steps:torch.save({'format':'silu_reference_v1','scene':a.scene,'beta':model.beta,'step':step,'reference':{k:v.cpu() for k,v in model.state_dict().items()},'GT_used':False},out/f'{a.scene}_silu_{step:06d}.pt')
(out/f'{a.scene}_silu_training.json').write_text(json.dumps({'train_samples':meta,'GT_used':False,'seed':934,'loss':'feature L1 + .5 DC display sensitivity L1 + temporal pair display sensitivity L1','mixed_precision':True,'logs':logs,'NPU_verified':False},indent=2))
