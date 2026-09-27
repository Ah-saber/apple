"""Search exact positive channel rescalings using train-only quantization loss.
No convolution coefficient function is learned; float64 function is preserved.
"""
import argparse,copy,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-BODY-DISTILL-20260927'
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from body_candidates import load_body,BASE_CASE,prepare_inputs
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)];rows=[rows[int(i)] for i in np.linspace(0,len(rows)-5,8,dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2791);rng=np.random.default_rng(2791);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base=load_body(a.scene,BASE_CASE,out);m=base.core.model;X=[];R=[];metadata=[];size=32
with torch.no_grad():
 for row in rows:
  features=[];refs=[];cy,cx,ch,cw=row['train_roi_tlhw']
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();x,ctx=prepare_inputs(base,x,ctx);feat=m.head(base.core.half_input(base.front(x)));features.append(feat);ref=m.global_reference.encode(base.core.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));refs.append(F.interpolate(ref,size=feat.shape[-2:],mode='bilinear',align_corners=False))
  for _ in range(2):
   y=cy//2+int(rng.integers(0,ch//2-size+1));x=cx//2+int(rng.integers(0,cw//2-size+1));X.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in features]));R.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in refs]));metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[y*2,x*2,size*2,size*2],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
X=torch.cat(X).to(memory_format=torch.channels_last);R=torch.cat(R);tail=copy.deepcopy(m.tail);projection=copy.deepcopy(base.output.conv);del base,m,features,refs,feat,ref,ctx;torch.cuda.empty_cache()
def quantize(body,mode):
 result=copy.deepcopy(body)
 if mode:
  with torch.no_grad():
   for conv in result:
    if isinstance(conv,nn.Conv2d):
     w=conv.weight.float();scale=(w.abs().amax((1,2,3),keepdim=True) if mode=='per_output' else w.abs().max()).clamp_min(1e-9)/127;conv.weight.copy_((w/scale).round().clamp(-127,127)*scale)
 return result
def channel_step(body,pair,channel,exponent):
 result=copy.deepcopy(body);first,second=result[pair*2],result[(pair+1)*2];scale=2.**exponent
 with torch.no_grad():
  changes=((first.weight[channel],first.weight[channel].double()*scale),(first.bias[channel:channel+1],first.bias[channel:channel+1].double()*scale),(second.weight[:,channel],second.weight[:,channel].double()/scale))
  # A rescaled coefficient must remain exactly representable in Half.
  for old,new in changes:
   cast=new.to(torch.float16)
   if not torch.isfinite(cast).all() or not torch.equal(cast.double(),new):return None
  for old,new in changes:old.copy_(new)
 return result
with torch.inference_mode():
 for depth in (3,2):
  teacher_case=('body3_trained' if a.scene=='ordinary' else 'body3_augtrained') if depth==3 else ('body2_trained' if a.scene=='ordinary' else 'body2_pretrained')
  original=copy.deepcopy(load_body(a.scene,teacher_case,out).core.model.body).eval();target=projection(tail(original(X)+R))[:,:9].float().clamp(0,1)[:,:,6:-6,6:-6]*255
  def objective(body):
   losses={}
   for mode in (None,'per_tensor','per_output'):
    model=quantize(body,mode);pred=projection(tail(model(X)+R))[:,:9].float().clamp(0,1)[:,:,6:-6,6:-6]*255;error=pred-target;pair=error.reshape(-1,2,9,size-12,size-12);temporal=(pair[:,1]-pair[:,0]).abs().mean();local=F.avg_pool2d(error,4,4).abs().mean();loss=error.abs().mean()+.1*error.square().mean()+2*temporal+2*local;losses['float' if mode is None else mode]=float(loss)
   score=max(losses['per_tensor'],losses['per_output'])+.25*(losses['per_tensor']+losses['per_output'])+losses['float']
   return score,losses
  body=copy.deepcopy(original);best,losses=objective(body);initial={'score':best,'losses':losses};logs=[];started=time.perf_counter();changes=[]
  for sweep in range(4):
   accepted=0
   for pair in range(depth-1):
    for channel in range(16):
     for exponent in (-1,1):
      candidate=channel_step(body,pair,channel,exponent)
      if candidate is None:continue
      score,new_losses=objective(candidate)
      if score<best-1e-5:
       body=candidate;best=score;losses=new_losses;accepted+=1;changes.append({'sweep':sweep,'pair':pair,'channel':channel,'power_of_two':exponent,'score':best});break
   item={'sweep':sweep,'accepted':accepted,'score':best,'losses':losses,'elapsed_s':time.perf_counter()-started};logs.append(item);print('SEARCH',a.scene,depth,item,flush=True)
   if not accepted:break
  # Real coefficient function checked on the actual trained weights in float64.
  xx=X[:2].double();proof=float((original.double()(xx)-body.double()(xx)).abs().max());assert proof<1e-10;body=body.half();path=out/f'{a.scene}_body{depth}_quantsearch.pt'
  if path.exists():raise FileExistsError(path)
  torch.save({'format':'body_positive_scaling_quantsearch_v1','scene':a.scene,'depth':depth,'body':{k:v.cpu() for k,v in body.state_dict().items()},'teacher_case':teacher_case,'float64_max_abs':proof,'GT_used':False},path)
  (out/f'{a.scene}_body{depth}_quantsearch.json').write_text(json.dumps({'scene':a.scene,'depth':depth,'teacher_case':teacher_case,'teacher_selected_using_prior_architecture_test_evidence':True,'calibration_split':'train','calibration_samples':metadata,'seed':2791,'GT_used':False,'test_used_for_quantization_search':False,'exact_half_coefficient_representability_enforced':True,'float64_max_abs':proof,'changes':changes,'initial':initial,'final':{'score':best,'losses':losses},'logs':logs,'SDK_activation_quantization_mode_not_simulated':True},indent=2));print('COMPLETE',a.scene,depth,flush=True)
