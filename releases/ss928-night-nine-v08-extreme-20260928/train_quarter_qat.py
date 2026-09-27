"""Weight-only 8-bit pressure training. SDK activation quantization is unverified."""
import argparse,copy,fcntl,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import load_deployment,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2801);rng=np.random.default_rng(2801);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
teacher=load_deployment(a.scene,'combo_input9_half_aligned16_nearest',out);teacher.requires_grad_(False);size=128;xs=[];ys=[];thumbs=[];grids=[];metadata=[]
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);value=teacher(x,ctx).float().clamp(0,255).round();raw=F.avg_pool2d(value,3,3)/255;pair.append((x,raw,ctx))
  for _ in range(6):
   y=int(rng.integers((cy+3)//4,(cy+ch-size)//4+1));xx=int(rng.integers((cx+3)//4,(cx+cw-size)//4+1));h=size//4;xs.append(torch.cat([v[0][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().half());ys.append(torch.cat([v[1][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().float());thumbs.append(torch.cat([v[2] for v in pair]).cpu().float());gy,gx=torch.meshgrid(torch.arange(y,y+h,dtype=torch.float32),torch.arange(xx,xx+h,dtype=torch.float32),indexing='ij');grids.append(torch.stack(((gx+.5)/320*2-1,(gy+.5)/256*2-1),-1).repeat(2,1,1,1));metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[4*y,4*xx,size,size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
xs=torch.stack(xs);ys=torch.stack(ys);thumbs=torch.stack(thumbs);grids=torch.stack(grids);del teacher,pair,x,ctx,raw,value;torch.cuda.empty_cache();count=len(metadata)-24;halo=20;valid=size-2*halo
source_case='cal_quarter_w32_3x3_body1';student=load_deployment(a.scene,source_case,out).float().requires_grad_(True);mode='none'
def fake_forward(layer,x):
 w=layer.weight
 if mode!='none':
  scale=(w.detach().abs().amax((1,2,3),keepdim=True) if mode=='po' else w.detach().abs().max()).clamp_min(1e-9)/127
  qw=(w/scale).round().clamp(-127,127)*scale;w=w+(qw-w).detach()
 return F.conv2d(x,w,layer.bias,layer.stride,layer.padding,layer.dilation,layer.groups)
for layer in student.modules():
 if isinstance(layer,nn.Conv2d):layer.forward=types.MethodType(fake_forward,layer)
params=list(student.parameters());opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8000,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter()
def forward(ids):
 xx=xs[ids].flatten(0,1).cuda().float();ctx=thumbs[ids].flatten(0,1).cuda();grid=grids[ids].flatten(0,1).cuda()
 with torch.autocast('cuda',dtype=torch.float16):
  v=student.body(student.front(xx));r=student.reference.encode(ctx);r=student.reference.project(r+r.mean((-2,-1),keepdim=True))
  r=F.grid_sample(r.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)
  return student.output.native(student.tail(v+r)).float().clamp(0,1)
for step in range(1,8001):
 mode=('po','pt','po','none')[(step-1)%4];ids=rng.integers(0,count,4);target=ys[ids].flatten(0,1).cuda();opt.zero_grad(set_to_none=True);pred=forward(ids);err=(pred-target)[:,:,halo:-halo,halo:-halo]*255;pe=err.reshape(-1,2,1,valid,valid);pixel=err.abs().mean();temporal=(pe[:,1]-pe[:,0]).abs().mean();td=(target[1::2]-target[::2]).abs()[:,:,halo:-halo,halo:-halo]*255;weak=(td>=.4)&(td<=8);we=((pe[:,1]-pe[:,0]).abs()*weak).sum()/weak.sum().clamp_min(1);local=F.avg_pool2d(err,8,8).abs().mean();grad=(err[:,:,1:]-err[:,:,:-1]).abs().mean()+(err[:,:,:,1:]-err[:,:,:,:-1]).abs().mean();loss=pixel+.8*temporal+we+.7*local+.2*grad
 scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,5);scaler.step(opt);scaler.update();scheduler.step()
 if step==1 or step%2000==0:
  held={}
  with torch.no_grad():
   for mode in ('none','po','pt'):held[mode]=float((forward(np.arange(count,count+2))-ys[count:count+2].flatten(0,1).cuda())[:,:,halo:-halo,halo:-halo].abs().mean()*255)
  row={'attempted_iteration':step,'loss':float(loss),'pixel_gray':float(pixel),'temporal_gray':float(temporal),'weak_gray':float(we),'heldout_two_pairs_gray':held,'elapsed_seconds':time.perf_counter()-started};logs.append(row);print('TRAIN',a.scene,json.dumps(row),flush=True)
checkpoint={'model':{n:v.detach().cpu() for n,v in student.state_dict().items()},'scene':a.scene,'source_case':source_case,'qat_attempted_iterations':8000,'successful_optimizer_updates':int(opt.state[next(iter(opt.state))]['step'].item()),'seed':2801,'weight_only_quantization_pressure':True,'activation_quantization':False,'SDK_verified':False,'GT_used':False,'test_used_for_training':False};torch.save(checkpoint,out/f'{a.scene}_quarter_w32_3x3_body1_qat008000.pt');(out/f'{a.scene}_quarter_qat_training.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k!='model'}|{'train_pairs':count,'heldout_pairs':24,'heldout_diagnostic_pairs':2,'samples':metadata,'logs':logs},indent=2));print('FINISHED',flush=True)
