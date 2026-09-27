"""Train quarter fused model with GT weak-motion loss, strict eval-label separation."""
import argparse,copy,fcntl,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2808);rng=np.random.default_rng(2808);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');evalds=dataset_for_config(state['config'],'test');ev=next(r for r in evalds.records if r['scene_id']=='night_'+a.scene);ep=Path(ev['target']['path']).parent;ty,tx,th,tw=ev['eval_crop_tlhw'];tc=120 if special else 60;records=[];excluded=[]
for row in ds.records:
 if row['scene_id']!='night_'+a.scene:continue
 cy,cx,ch,cw=row['train_roi_tlhw'];overlap=max(cy,ty)<min(cy+ch,ty+th) and max(cx,tx)<min(cx+cw,tx+tw);bad=Path(row['target']['path']).parent==ep and max(0,row['frame_id']-1)<tc and overlap
 if bad:excluded.append(row['sample_id'])
 else:records.append(row)
assert len(records)>=8;rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
teacher=load_deployment(a.scene,'body3_quantsearch',out).requires_grad_(False);student=load_deployment(a.scene,'quarter_w32_3x3_body1_stage1_fusedtail',out).float();student.reference.requires_grad_(False);refmodels=[student.reference,copy.deepcopy(student.reference),copy.deepcopy(student.reference)]
for mode,reference in enumerate(refmodels[1:]):
 with torch.no_grad():
  for layer in reference.modules():
   if isinstance(layer,nn.Conv2d):
    w=layer.weight;scale=(w.abs().amax((1,2,3),keepdim=True) if mode==0 else w.abs().max()).clamp_min(1e-9)/127;layer.weight.copy_(((w/scale).round().clamp(-127,127)*scale).half().float())
size=128;xs=[];ys=[];gts=[];refs=[];metadata=[]
with torch.inference_mode():
 for row in rows:
  cy,cx,ch,cw=row['train_roi_tlhw'];pair=[]
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(teacher,x,ctx);value=teacher(x,ctx).float().clamp(0,255).round();raw=F.avg_pool2d(value,3,3)/255;path=(Path(state['config']['data_root'])/rr['target']['path']).with_name(f'{frame:06d}.png');gt=torch.from_numpy(np.asarray(Image.open(path),dtype=np.float32)/255)[None,None].cuda();assert gt.shape==(1,1,1024,1280);refvalues=[]
   with torch.autocast('cuda',dtype=torch.float16):
    for reference in refmodels:
     r=reference.encode(ctx);r=reference.project(r+r.mean((-2,-1),keepdim=True));refvalues.append(F.interpolate(r,size=(256,320),mode='bilinear',align_corners=False))
   pair.append((x,raw,gt,refvalues))
  for _ in range(6):
   y=int(rng.integers((cy+3)//4,(cy+ch-size)//4+1));xx=int(rng.integers((cx+3)//4,(cx+cw-size)//4+1));h=size//4;xs.append(torch.cat([v[0][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().half());ys.append(torch.cat([v[1][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().float());gts.append(torch.cat([v[2][:,:,4*y:4*y+size,4*xx:4*xx+size] for v in pair]).cpu().float());refs.append(torch.stack([torch.cat([v[3][mode][:,:,y:y+h,xx:xx+h] for v in pair]).cpu().half() for mode in range(3)]));metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[4*y,4*xx,size,size],'train_roi':row['train_roi_tlhw'],'GT_parent':str(Path(row['target']['path']).parent)})
  print('CACHE',a.scene,row['frame_id'],flush=True)
xs=torch.stack(xs);ys=torch.stack(ys);gts=torch.stack(gts);refs=torch.stack(refs);del teacher,refmodels,pair,x,ctx,raw,gt,value,r,refvalues;torch.cuda.empty_cache();count=len(metadata)-24;halo=20;valid=size-2*halo;mode='none'
def fake_forward(layer,x):
 w=layer.weight
 if mode!='none':
  scale=(w.detach().abs().amax((1,2,3),keepdim=True) if mode=='po' else w.detach().abs().max()).clamp_min(1e-9)/127;qw=(w/scale).round().clamp(-127,127)*scale;w=w+(qw-w).detach()
 return F.conv2d(x,w,layer.bias,layer.stride,layer.padding,layer.dilation,layer.groups)
for sub in (student.front,student.body,student.output):
 for layer in sub.modules():
  if isinstance(layer,nn.Conv2d):layer.forward=types.MethodType(fake_forward,layer)
params=[v for v in student.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=4e-5,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,16000,eta_min=2e-6);scaler=torch.amp.GradScaler('cuda');logs=[];started=time.perf_counter()
def forward(ids):
 xx=xs[ids].flatten(0,1).cuda().float();index=0 if mode=='none' else 1 if mode=='po' else 2;r=refs[ids,index].flatten(0,1).cuda().float()
 with torch.autocast('cuda',dtype=torch.float16):return student.output.native(student.tail(student.body(student.front(xx))+r)).float().clamp(0,1)
for step in range(1,16001):
 mode=('none','none','none','po','none','none','none','pt')[(step-1)%8];ids=rng.integers(0,count,4);target=ys[ids].flatten(0,1).cuda();gt=gts[ids].flatten(0,1).cuda();opt.zero_grad(set_to_none=True);pred=forward(ids);teacher_error=(pred-target)[:,:,halo:-halo,halo:-halo]*255;gt_error=(pred-gt)[:,:,halo:-halo,halo:-halo]*255;pe=gt_error.reshape(-1,2,1,valid,valid);te=teacher_error.reshape(-1,2,1,valid,valid);pd=(pred[1::2]-pred[::2])[:,:,halo:-halo,halo:-halo]*255;gd=(gt[1::2]-gt[::2])[:,:,halo:-halo,halo:-halo]*255;weak=(gd.abs()>=.4)&(gd.abs()<=8);we=((pd-gd).abs()*weak).sum()/weak.sum().clamp_min(1);temporal=(pe[:,1]-pe[:,0]).abs().mean();pixel=gt_error.abs().mean();local=F.avg_pool2d(gt_error,8,8).abs().mean();grad=(gt_error[:,:,1:]-gt_error[:,:,:-1]).abs().mean()+(gt_error[:,:,:,1:]-gt_error[:,:,:,:-1]).abs().mean();loss=.5*pixel+.5*teacher_error.abs().mean()+2*temporal+4*we+.5*(te[:,1]-te[:,0]).abs().mean()+.5*local+.2*grad
 scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,5);scaler.step(opt);scaler.update();scheduler.step()
 if step==1 or step%2000==0:
  held={}
  with torch.no_grad():
   for mode in ('none','po','pt'):held[mode]=float((forward(np.arange(count,count+2))-gts[count:count+2].flatten(0,1).cuda())[:,:,halo:-halo,halo:-halo].abs().mean()*255)
  row={'attempted_iteration':step,'loss':float(loss),'pixel_GT_gray':float(pixel),'temporal_GT_gray':float(temporal),'weak_GT_gray':float(we),'heldout_two_pairs_GT_gray':held,'elapsed_seconds':time.perf_counter()-started};logs.append(row);print('TRAIN',a.scene,json.dumps(row),flush=True)
checkpoint={'model':{n:v.detach().cpu() for n,v in student.state_dict().items()},'scene':a.scene,'source_case':'quarter_w32_3x3_body1_stage1_fusedtail','GT_used':True,'new_attempted_iterations':16000,'successful_optimizer_updates':int(opt.state[next(iter(opt.state))]['step'].item()),'seed':2808,'weight_pressure_fraction':.25,'activation_quantization':False,'test_GT_labels_excluded_from_training_ROIs':True,'test_used_for_training':False};torch.save(checkpoint,out/f'{a.scene}_quarter_w32_3x3_body1_gtweak016000.pt');(out/f'{a.scene}_gtweak_training.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k!='model'}|{'train_pairs':count,'heldout_pairs':24,'heldout_diagnostic_pairs':2,'eval_GT_parent':str(ep),'eval_ROI':ev['eval_crop_tlhw'],'excluded_records':excluded,'samples':metadata,'logs':logs},indent=2));print('FINISHED',flush=True)
