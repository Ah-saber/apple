"""Teacher distillation of 4x16-channel body to 1/2/3 convolutions.
Train ROI only; fixed 24000 additional steps (32000 total); test/GT excluded from optimization.
"""
import argparse,fcntl,json,sys,time,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-BODY-DISTILL-20260927';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from body_candidates import load_body,ShortBody,BASE_CASE,prepare_inputs
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(2784);rng=np.random.default_rng(2784);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
teacher=load_body(a.scene,BASE_CASE,out);teacher.requires_grad_(False);m=teacher.core.model;size=64;cache=[];metadata=[]
with torch.no_grad():
 for row in rows:
  frames=[];cy,cx,ch,cw=row['train_roi_tlhw'];assert cy%2==cx%2==0
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx)
   features=[m.head(teacher.core.half_input(teacher.front(x)))]
   preactivations=[]
   for layer in m.body:
    assert isinstance(layer.norm,torch.nn.Identity) and isinstance(layer.act,torch.nn.ReLU)
    pre=layer.conv1(features[-1]);preactivations.append(pre);features.append(layer.act(pre))
   assert torch.equal(features[-1],m.body(features[0]))
   ref=m.global_reference.encode(teacher.core.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));ref=F.interpolate(ref,size=features[-1].shape[-2:],mode='bilinear',align_corners=False)
   frames.append((features,ref,preactivations))
  for _ in range(6):
   y=cy//2+int(rng.integers(0,ch//2-size+1));x=cx//2+int(rng.integers(0,cw//2-size+1))
   cache.append([torch.cat([fs[k][:,:,y:y+size,x:x+size] for fs,ref,pre in frames]).cpu() for k in range(5)]+[torch.cat([ref[:,:,y:y+size,x:x+size] for fs,ref,pre in frames]).cpu()]+[torch.cat([pre[k][:,:,y:y+size,x:x+size] for fs,ref,pre in frames]).cpu() for k in range(4)])
   metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[y*2,x*2,size*2,size*2],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
arrays=[torch.stack([item[k] for item in cache]) for k in range(10)];del cache
original_convs=[copy.deepcopy(layer.conv1.rep_conv).cpu() for layer in m.body]
teacher_body=copy.deepcopy(m.body).float().requires_grad_(False)
head_bias=m.head[0].bias.detach().float().view(1,16,1,1);head_dc=m.head[0].weight.detach().float().sum((1,2,3)).view(1,16,1,1)
tail=copy.deepcopy(m.tail).float().requires_grad_(False);projection=copy.deepcopy(teacher.output.conv).float().requires_grad_(False)
del teacher,frames,features,ref,x,ctx;m=None;torch.cuda.empty_cache()
train_count=len(metadata)-24;assert train_count>0
# Approximate skipped teacher blocks by fitting each surviving convolution.
# The normal equations only use the training subset and interior points.
def initialize(student,endpoints):
 previous=0
 with torch.no_grad():
  for index,end in enumerate(endpoints):
   if end==previous+1:
    student[index*2].load_state_dict(original_convs[previous].state_dict());previous=end;continue
   xtx=torch.zeros(145,145,dtype=torch.float64,device='cuda');xty=torch.zeros(145,16,dtype=torch.float64,device='cuda')
   for sample in range(train_count):
    xx=arrays[previous][sample].float().cuda();yy=arrays[6+end-1][sample].float().cuda();patch=F.unfold(xx,3,padding=1).reshape(2,144,size,size)[:,:,6:-6,6:-6].flatten(2).transpose(1,2)[:,::17].reshape(-1,144).double();tar=yy[:,:,6:-6,6:-6].flatten(2).transpose(1,2)[:,::17].reshape(-1,16).double();design=torch.cat((patch,torch.ones_like(patch[:,:1])),1);xtx+=design.T@design;xty+=design.T@tar
   ridge=xtx.diagonal().mean()*1e-5;fit=torch.linalg.solve(xtx+torch.eye(145,device='cuda',dtype=torch.float64)*ridge,xty);conv=student[index*2];conv.weight.copy_(fit[:144].T.reshape(16,16,3,3));conv.bias.copy_(fit[-1]);previous=end
for depth,endpoints in ((3,[1,2,4]),(2,[2,4])):
 student=ShortBody(depth,device='cuda').to(memory_format=torch.channels_last);checkpoint=torch.load(out/f'{a.scene}_body{depth}_pre008000.pt',map_location='cpu',weights_only=True);student.load_state_dict(checkpoint['body'],strict=True);initial={k:v.detach().cpu().clone() for k,v in student.state_dict().items()};opt=torch.optim.AdamW(student.parameters(),lr=2e-5,weight_decay=0);scaler=torch.amp.GradScaler('cuda');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24000,eta_min=1e-6);local_rng=np.random.default_rng(2784);logs=[];started=time.perf_counter();halo=6
 def loss_for(ids,training):
  xx=arrays[0][ids].reshape(-1,16,size,size).cuda().float();target=arrays[4][ids].reshape(-1,16,size,size).cuda().float();ref=arrays[5][ids].reshape(-1,16,size,size).cuda().float()
  if training:
   paired_count=len(ids);apply=(torch.rand(paired_count,1,1,1,device='cuda')<.7).repeat_interleave(2,0);gain=(.85+.30*torch.rand(paired_count,1,1,1,device='cuda')).repeat_interleave(2,0);offset=((torch.rand(paired_count,1,1,1,device='cuda')-.5)*.06).repeat_interleave(2,0);xx=torch.where(apply,(xx-head_bias)*gain+head_bias+head_dc*offset,xx)
  with torch.autocast('cuda',dtype=torch.float16):
   with torch.no_grad():target=teacher_body(xx);target_phase=projection(tail(target+ref))[:,:9]
   pred=student(xx);phase=projection(tail(pred+ref))[:,:9]
  error=(phase.float().clamp(0,1)-target_phase.float().clamp(0,1))[:,:,halo:-halo,halo:-halo]*255;pair=error.reshape(-1,2,9,size-2*halo,size-2*halo);temporal=(pair[:,1]-pair[:,0]).abs().mean();local_mean=F.avg_pool2d(error,8,8).abs().mean();feature=((pred.float()-target)[:,:,halo:-halo,halo:-halo]).abs().mean()/target[:,:,halo:-halo,halo:-halo].abs().mean().clamp_min(.01);gradient=.5*((error[:,:,1:]-error[:,:,:-1]).abs().mean()+(error[:,:,:,1:]-error[:,:,:,:-1]).abs().mean());teacher_pair=target_phase.float().clamp(0,1)[:,:,halo:-halo,halo:-halo].reshape(-1,2,9,size-2*halo,size-2*halo)*255;dt=(teacher_pair[:,1]-teacher_pair[:,0]).abs();mask=(dt>=.5)&(dt<=8);weak=(pair[:,1]-pair[:,0])[mask].abs().mean() if mask.any() else temporal;loss=error.abs().mean()+.25*error.square().mean()+2*temporal+2*weak+2*local_mean+.5*gradient+.1*feature
  return loss,{'phase_mae_gray':float(error.abs().mean().detach()),'temporal_mae_gray':float(temporal.detach()),'local_mean_mae_gray':float(local_mean.detach()),'feature_relative_mae':float(feature.detach())}
 for step in range(1,24001):
  ids=local_rng.integers(0,train_count,2);opt.zero_grad(set_to_none=True);loss,info=loss_for(ids,True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),1);scaler.step(opt);scaler.update();scheduler.step()
  if step%1000==0:
   with torch.no_grad():_,holdout=loss_for(np.arange(train_count,train_count+2),False)
   item={'step':step,'loss':float(loss.detach()),**info,'train_roi_holdout':holdout,'elapsed_s':time.perf_counter()-started};logs.append(item);print('TRAIN',a.scene,depth,item,flush=True)
 path=out/f'{a.scene}_body{depth}_aug032000.pt'
 if path.exists():raise FileExistsError(path)
 torch.save({'format':'body_teacher_augmented_v3','scene':a.scene,'depth':depth,'step':32000,'body':{k:v.detach().cpu() for k,v in student.state_dict().items()},'initial_body':initial,'teacher_case':BASE_CASE,'GT_used':False},path)
 (out/f'{a.scene}_body{depth}_aug_training.json').write_text(json.dumps({'scene':a.scene,'depth':depth,'seed':2784,'steps':32000,'additional_steps':24000,'train_samples':metadata[:train_count],'train_roi_holdout_samples':metadata[train_count:],'GT_used':False,'test_used_for_training_or_checkpoint_selection':False,'teacher_case':BASE_CASE,'halo_feature_pixels':halo,'initialization':'fixed 8000-step preactivation checkpoint; no test checkpoint selection','augmentation':'70% pairs: front-like gain .85..1.15 and offset -.03...03 applied through frozen head DC response; shared between temporal pair; frozen teacher body queried on augmented features at each step','loss':'phase L1 + .25 MSE + 2 temporal + 2 teacher weak temporal + 2 local mean + .5 gradient + .1 relative feature L1','logs':logs,'NPU_verified':False},indent=2));print('COMPLETE',a.scene,depth,flush=True)
