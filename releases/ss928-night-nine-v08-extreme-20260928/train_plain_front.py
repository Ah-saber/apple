"""Direct nine-frame front distillation; train ROI only, fixed 8000 updates."""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',choices=['ordinary','special'],required=True);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from extreme_candidates import load_extreme,PlainFront,BASE_CASE,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.manual_seed(2794);rng=np.random.default_rng(2794);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
teacher=load_extreme(a.scene,BASE_CASE,out);teacher.requires_grad_(False);m=teacher.core.model
head=copy.deepcopy(m.head).float();body=copy.deepcopy(m.body).float();tail=copy.deepcopy(m.tail).float();projection=copy.deepcopy(teacher.output.conv).float()
for module in (head,body,tail,projection):module.requires_grad_(False)
size=96;inputs=[];targets=[];references=[];metadata=[]
with torch.inference_mode():
 for row in rows:
  pair=[];cy,cx,ch,cw=row['train_roi_tlhw'];assert cy%2==cx%2==0
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);target=teacher.front(x)
   ref=m.global_reference.encode(teacher.core.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));ref=F.interpolate(ref,size=(512,640),mode='bilinear',align_corners=False)
   pair.append((x,target,ref))
  for _ in range(6):
   y=cy//2+int(rng.integers(0,(ch-size)//2+1));xx=cx//2+int(rng.integers(0,(cw-size)//2+1));h=size//2
   inputs.append(torch.cat([v[0][:,:,2*y:2*y+size,2*xx:2*xx+size] for v in pair]).cpu().half())
   targets.append(torch.cat([v[1][:,:,y:y+h,xx:xx+h] for v in pair]).cpu().float())
   references.append(torch.cat([v[2][:,:,y:y+h,xx:xx+h] for v in pair]).cpu().half())
   metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[2*y,2*xx,size,size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
inputs=torch.stack(inputs);targets=torch.stack(targets);references=torch.stack(references);del teacher,m,pair,x,ctx,target,ref;torch.cuda.empty_cache();train_count=len(metadata)-24;assert train_count>0
halo=8;valid=size//2-2*halo
for k,last in ((3,1),(3,3),(5,1)):
 tag=f'front{k}x{last}';student=PlainFront(k,last).cuda().to(memory_format=torch.channels_last);initial={n:v.detach().cpu().clone() for n,v in student.state_dict().items()};opt=torch.optim.AdamW(student.parameters(),lr=1e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8000,eta_min=2e-6);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter();local_rng=np.random.default_rng(2794)
 def loss_for(ids):
  xx=inputs[ids].flatten(0,1).cuda().float();target=targets[ids].flatten(0,1).cuda();ref=references[ids].flatten(0,1).cuda().float()
  with torch.autocast('cuda',dtype=torch.float16):
   pred=student(xx);phase=projection(tail(body(head(pred))+ref))[:,:9]
   with torch.no_grad():tar=projection(tail(body(head(target))+ref))[:,:9]
  err=(phase.float().clamp(0,1)-tar.float().clamp(0,1))[:,:,halo:-halo,halo:-halo]*255
  pe=err.reshape(-1,2,9,valid,valid);temporal=(pe[:,1]-pe[:,0]).abs().mean();front=(pred-target)[:,:,halo:-halo,halo:-halo]*255
  target_diff=(tar.float()[1::2]-tar.float()[::2]).abs()[:,:,halo:-halo,halo:-halo]*255;weak=(target_diff>=.4)&(target_diff<=8)
  weak_temporal=((pe[:,1]-pe[:,0]).abs()*weak).sum()/weak.sum().clamp_min(1)
  pixel=err.abs().mean();local=F.avg_pool2d(err,8,8).abs().mean();feature=front.abs().mean();loss=pixel+.6*temporal+.7*weak_temporal+.4*local+.2*feature
  return loss,{'loss':float(loss),'phase_gray':float(pixel),'temporal_gray':float(temporal),'weak_temporal_gray':float(weak_temporal),'front_gray':float(feature)}
 for step in range(1,8001):
  ids=local_rng.integers(0,train_count,4);opt.zero_grad(set_to_none=True);loss,row=loss_for(ids);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),5);scaler.step(opt);scaler.update();scheduler.step()
  if step==1 or step%1000==0:
   with torch.no_grad():_,held=loss_for(np.arange(train_count,train_count+2))
   row.update(step=step,heldout_two_pairs=held,elapsed_seconds=time.perf_counter()-started);logs.append(row);print('TRAIN',a.scene,tag,json.dumps(row),flush=True)
 checkpoint={'front':{n:v.detach().cpu() for n,v in student.state_dict().items()},'initial_front':initial,'scene':a.scene,'tag':tag,'step':8000,'seed':2794,'train_samples':metadata[:train_count],'heldout_samples':metadata[train_count:],'GT_used':False,'test_used_for_training':False}
 torch.save(checkpoint,out/f'{a.scene}_{tag}_008000.pt');(out/f'{a.scene}_{tag}_training.json').write_text(json.dumps({'scene':a.scene,'tag':tag,'step':8000,'seed':2794,'train_pairs':train_count,'heldout_pairs':24,'heldout_diagnostic_pairs':2,'GT_used':False,'test_used_for_training':False,'teacher_case':BASE_CASE,'logs':logs,'samples':metadata},indent=2));del student,opt;torch.cuda.empty_cache()
print('FINISHED',a.scene,flush=True)
