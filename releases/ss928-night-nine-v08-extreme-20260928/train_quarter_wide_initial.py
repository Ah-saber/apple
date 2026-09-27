"""Quarter-grid student with frozen multi-scale reference encoder, native gray distillation."""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from quarter_candidates import QuarterSystem,QuarterFront,load_extreme,BASE_CASE,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2798);rng=np.random.default_rng(2798);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
teacher=load_extreme(a.scene,BASE_CASE,out);teacher.requires_grad_(False);m=teacher.core.model;reference=copy.deepcopy(m.global_reference).cpu().float();size=128;xs=[];ys=[];refs=[];metadata=[]
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);value=teacher(x,ctx).float().clamp(0,255).round();raw=F.avg_pool2d(value,3,3)/255;r=m.global_reference.encode(teacher.core.half_input(ctx));r=m.global_reference.project(r+r.mean((-2,-1),keepdim=True));r=F.interpolate(r,size=(256,320),mode='bilinear',align_corners=False);pair.append((x,raw,r))
  for _ in range(6):
   y=int(rng.integers((cy+3)//4,(cy+ch-size)//4+1));xx=int(rng.integers((cx+3)//4,(cx+cw-size)//4+1));h=size//4;xs.append(torch.cat([v[0][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().half());ys.append(torch.cat([v[1][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().float());refs.append(torch.cat([v[2][:,:,y:y+h,xx:xx+h] for v in pair]).cpu().half());metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[4*y,4*xx,size,size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
xs=torch.stack(xs);ys=torch.stack(ys);refs=torch.stack(refs);del teacher,m,pair,x,ctx,raw,value,r;torch.cuda.empty_cache();count=len(metadata)-24;halo=20;valid=size-2*halo
for kind,depth in (('k4',1),('3x3',1)):
 torch.manual_seed(2798);local_rng=np.random.default_rng(2798);student=QuarterSystem(reference,depth,'pixel',kind,32).cuda().float().to(memory_format=torch.channels_last);student.reference.requires_grad_(False);transform=nn.Conv2d(16,32,1).cuda();nn.init.zeros_(transform.weight);nn.init.zeros_(transform.bias)
 params=list(student.front.parameters())+list(student.body.parameters())+list(student.tail.parameters())+list(student.output.parameters())+list(transform.parameters());opt=torch.optim.AdamW(params,lr=2e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24000,eta_min=2e-6);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter()
 def forward(ids):
  xx=xs[ids].flatten(0,1).cuda().float();r=refs[ids].flatten(0,1).cuda().float()
  with torch.autocast('cuda',dtype=torch.float16):return student.output.native(student.tail(student.body(student.front(xx))+transform(r))).float().clamp(0,1)
 for step in range(1,16001):
  ids=local_rng.integers(0,count,4);target=ys[ids].flatten(0,1).cuda();opt.zero_grad(set_to_none=True);pred=forward(ids);err=(pred-target)[:,:,halo:-halo,halo:-halo]*255;pe=err.reshape(-1,2,1,valid,valid);pixel=err.abs().mean();temporal=(pe[:,1]-pe[:,0]).abs().mean();td=(target[1::2]-target[::2]).abs()[:,:,halo:-halo,halo:-halo]*255;weak=(td>=.4)&(td<=8);we=((pe[:,1]-pe[:,0]).abs()*weak).sum()/weak.sum().clamp_min(1);local=F.avg_pool2d(err,8,8).abs().mean();grad=(err[:,:,1:]-err[:,:,:-1]).abs().mean()+(err[:,:,:,1:]-err[:,:,:,:-1]).abs().mean();loss=pixel+.8*temporal+we+.7*local+.2*grad
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,5);scaler.step(opt);scaler.update();scheduler.step()
  if step==1 or step%2000==0:
   with torch.no_grad():held=forward(np.arange(count,count+2));hy=ys[count:count+2].flatten(0,1).cuda();hm=float((held-hy)[:,:,halo:-halo,halo:-halo].abs().mean()*255)
   row={'step':step,'loss':float(loss),'pixel_gray':float(pixel),'temporal_gray':float(temporal),'weak_temporal_gray':float(we),'heldout_two_pairs_mae_gray':hm,'elapsed_seconds':time.perf_counter()-started};logs.append(row);print('TRAIN',a.scene,kind,depth,json.dumps(row),flush=True)
 # Fold the learned reference transform into the original 64x64 projection.
 with torch.no_grad():
  old=student.reference.project;w=reference.project.weight.detach().cuda().double();b=reference.project.bias.detach().cuda().double();matrix=transform.weight.detach().double()[:,:,0,0];bias=transform.bias.detach().double();old.weight.copy_(torch.einsum('oi,ickl->ockl',matrix,w));old.bias.copy_(matrix@b+bias)
 checkpoint={'model':{n:v.detach().cpu() for n,v in student.state_dict().items()},'reference_transform':transform.state_dict(),'scene':a.scene,'front_kind':kind,'depth':depth,'step':24000,'successful_optimizer_updates':int(opt.state[next(iter(opt.state))]['step'].item()),'seed':2798,'GT_used':False,'test_used':False};torch.save(checkpoint,out/f'{a.scene}_quarter_w32_{kind}_body{depth}_024000.pt');(out/f'{a.scene}_quarter_w32_{kind}_body{depth}_training.json').write_text(json.dumps({'scene':a.scene,'front_kind':kind,'depth':depth,'step':24000,'successful_optimizer_updates':int(opt.state[next(iter(opt.state))]['step'].item()),'seed':2798,'train_pairs':count,'heldout_pairs':24,'heldout_diagnostic_pairs':2,'GT_used':False,'test_used_for_training':False,'reference_transform_folded_before_resize_at_export':True,'samples':metadata,'logs':logs},indent=2));del student,opt,transform;torch.cuda.empty_cache()
print('FINISHED',flush=True)
