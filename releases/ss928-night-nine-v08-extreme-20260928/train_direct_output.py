"""Train a native-gray kernel4 stride2 output; preserve full output at inference."""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from extreme_candidates import load_extreme,DirectOutput4,NativeOutput,BASE_CASE,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2796);rng=np.random.default_rng(2796);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
teacher=load_extreme(a.scene,BASE_CASE,out);teacher.requires_grad_(False);m=teacher.core.model;student=DirectOutput4().cuda();native=NativeOutput(teacher.output)
with torch.no_grad():
 w=native.conv.weight.float();kernel=torch.zeros(16,1,6,6,device='cuda')
 for py in range(2):
  for px in range(2):
   for ky in range(3):
    for kx in range(3):kernel[:,0,(2-ky)*2+py,(2-kx)*2+px]=w[py*2+px,:,ky,kx]
 student.conv.weight.copy_(kernel[:,:,1:-1,1:-1]);student.conv.bias.copy_(native.conv.bias.float().mean())
initial={n:v.detach().cpu().clone() for n,v in student.state_dict().items()};size=64;xs=[];ys=[];metadata=[]
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);v=m.body(m.head(teacher.core.half_input(teacher.front(x))));r=m.global_reference.encode(teacher.core.half_input(ctx));r=m.global_reference.project(r+r.mean((-2,-1),keepdim=True));v=m.tail(v+F.interpolate(r,size=(512,640),mode='bilinear',align_corners=False));raw=F.avg_pool2d(teacher.output(v).float().clamp(0,255).round(),3,3)/255;pair.append((v,raw))
  for _ in range(6):
   y=cy//2+int(rng.integers(0,ch//2-size+1));xx=cx//2+int(rng.integers(0,cw//2-size+1));xs.append(torch.cat([v[:,:,y:y+size,xx:xx+size] for v,t in pair]).cpu().half());ys.append(torch.cat([t[:,:,2*y:2*y+2*size,2*xx:2*xx+2*size] for v,t in pair]).cpu().float());metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[2*y,2*xx,2*size,2*size],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
xs=torch.stack(xs);ys=torch.stack(ys);del teacher,m,pair,x,ctx,v,r,raw,native;torch.cuda.empty_cache();train_count=len(metadata)-24;student=student.to(memory_format=torch.channels_last);opt=torch.optim.AdamW(student.parameters(),lr=5e-5,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8000,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter();halo=6
for step in range(1,8001):
 ids=rng.integers(0,train_count,4);xx=xs[ids].flatten(0,1).cuda().float();target=ys[ids].flatten(0,1).cuda();opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):pred=student.native(xx)
 err=(pred.float().clamp(0,1)-target)[:,:,halo:-halo,halo:-halo]*255;pe=err.reshape(-1,2,1,2*size-2*halo,2*size-2*halo);pixel=err.abs().mean();temporal=(pe[:,1]-pe[:,0]).abs().mean();local=F.avg_pool2d(err,8,8).abs().mean();td=(target[1::2]-target[::2]).abs()[:,:,halo:-halo,halo:-halo]*255;weak=(td>=.4)&(td<=8);we=((pe[:,1]-pe[:,0]).abs()*weak).sum()/weak.sum().clamp_min(1);loss=pixel+.7*temporal+.8*we+.5*local
 scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),5);scaler.step(opt);scaler.update();scheduler.step()
 if step==1 or step%1000==0:
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):held=student.native(xs[train_count:train_count+2].flatten(0,1).cuda().float()).float().clamp(0,1);hy=ys[train_count:train_count+2].flatten(0,1).cuda();hm=float((held-hy)[:,:,halo:-halo,halo:-halo].abs().mean()*255)
  row={'step':step,'loss':float(loss),'pixel_gray':float(pixel),'temporal_gray':float(temporal),'weak_temporal_gray':float(we),'heldout_two_pairs_mae_gray':hm,'elapsed_seconds':time.perf_counter()-started};logs.append(row);print('TRAIN',a.scene,json.dumps(row),flush=True)
torch.save({'output':{n:v.detach().cpu() for n,v in student.state_dict().items()},'initial_output':initial,'scene':a.scene,'step':8000,'seed':2796,'GT_used':False,'test_used':False},out/f'{a.scene}_direct4_008000.pt');(out/f'{a.scene}_direct4_training.json').write_text(json.dumps({'scene':a.scene,'step':8000,'seed':2796,'train_pairs':train_count,'heldout_pairs':24,'heldout_diagnostic_pairs':2,'GT_used':False,'test_used_for_training':False,'supervision':'rounded full parent output mean in 3x3 cells; native gray detail only','samples':metadata,'logs':logs},indent=2));print('FINISHED',flush=True)
