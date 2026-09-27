"""Train-only function-preserving power-of-two channel calibration of active graph."""
import argparse,copy,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--case',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment as load_extreme
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2799);rng=np.random.default_rng(2799);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-5,8,dtype=int)]
model=load_extreme(a.scene,a.case,out).float();quarter=not hasattr(model,'core');m=model if quarter else model.core.model;reference=m.reference if quarter else m.global_reference;body=[]
for layer in m.body:
 if isinstance(layer,nn.Conv2d):body.append(layer)
 elif hasattr(layer,'conv1'):body.append(layer.conv1.rep_conv)
# Every edge is exclusive, except the final body/ref addition, scaled together.
edges=[('front_internal',model.front.first,model.front.last,None)]
if getattr(model.front,'temporal_conv',None) is not None:edges.insert(0,('temporal_mix_front',model.front.temporal_conv,model.front.first,None))
if quarter:edges += [('front_body',model.front.last,body[0],None)]
else:edges += [('front_head',model.front.last,m.head[0],None),('head_body',m.head[0],body[0],None)]
edges += [(f'body_{i}',x,y,None) for i,(x,y) in enumerate(zip(body,body[1:]))]
tails=[v for v in m.tail.modules() if isinstance(v,nn.Conv2d)]
if len(tails)==1:
 tail_conv=tails[0];edges += [('body_reference_tail',body[-1],tail_conv,reference.project),('tail_projection',tail_conv,model.output.conv,None)]
elif quarter and not tails:
 if getattr(model,'refsplit',False):edges += [('body_projection',body[-1],model.output.conv,None),('reference_embed_phase',model.reference_embed,model.reference_phase,None)]
 else:edges += [('body_reference_projection',body[-1],model.output.conv,reference.project)]
else:raise ValueError('Unsupported tail geometry')
qmodels=[copy.deepcopy(model),copy.deepcopy(model)]
def quantize(dest,source,mode):
 with torch.no_grad():
  for d,s in zip(dest.modules(),source.modules()):
   if isinstance(s,nn.Conv2d):
    w=s.weight;scale=(w.abs().amax((1,2,3),keepdim=True) if mode==0 else w.abs().max()).clamp_min(1e-9)/127;d.weight.copy_(((w/scale).round().clamp(-127,127)*scale).half().float());
    if s.bias is not None:d.bias.copy_(s.bias)
for mode,q in enumerate(qmodels):quantize(q,model,mode)
xs=[];contexts=[];grids=[];samples=[];size=128 if quarter else 96;factor=4 if quarter else 2;feature_h=1024//factor;feature_w=1280//factor;halo=4 if quarter else 8;phase_count=16 if quarter else 4
for row in rows:
 cy,cx,ch,cw=row['train_roi_tlhw'];y=int(rng.integers((cy+factor-1)//factor,(cy+ch-size)//factor+1));xx=int(rng.integers((cx+factor-1)//factor,(cx+cw-size)//factor+1));paired=[];cc=[]
 for frame in (max(0,row['frame_id']-1),row['frame_id']):
  rr=dict(row,frame_id=frame);full=ds.normalized_stack(rr,(0,0,1024,1280));x=full[:,factor*y:factor*y+size,factor*xx:factor*xx+size][None].cuda().half().float();context,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));paired.append(x);cc.append(context[None].cuda().float())
 h=size//factor;yy=(torch.arange(y,y+h,device='cuda',dtype=torch.float32)+.5)*2/feature_h-1;xxx=(torch.arange(xx,xx+h,device='cuda',dtype=torch.float32)+.5)*2/feature_w-1;gy,gx=torch.meshgrid(yy,xxx,indexing='ij');grid=torch.stack((gx,gy),-1)[None].repeat(2,1,1,1);xs.append(torch.cat(paired));contexts.append(torch.cat(cc));grids.append(grid);samples.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[factor*y,factor*xx,size,size],'train_roi':row['train_roi_tlhw']})
x=torch.cat(xs);context=torch.cat(contexts);grid=torch.cat(grids);encoded=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
 for candidate in [model]+qmodels:
  owner=candidate if quarter else candidate.core.model;ref=owner.reference if quarter else owner.global_reference;v=ref.encode(context.half());encoded.append(v+v.mean((-2,-1),keepdim=True))
def phases(candidate,enc,xx=x,gg=grid,amp=True):
 cm=candidate if quarter else candidate.core.model
 with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
  v=cm.body(candidate.front(xx)) if quarter else cm.body(cm.head(candidate.front(xx)));ref=cm.reference if quarter else cm.global_reference
  if getattr(candidate,'refsplit',False):
   r=candidate.reference_embed(enc);r=F.grid_sample(r.to(dtype=gg.dtype),gg,mode='bilinear',padding_mode='border',align_corners=False).to(dtype=v.dtype);p=candidate.output.conv(v)+candidate.reference_phase(r)
  else:
   r=ref.project(enc);r=F.grid_sample(r.to(dtype=gg.dtype),gg,mode='bilinear',padding_mode='border',align_corners=False).to(dtype=v.dtype);p=candidate.output.conv(cm.tail(v+r))
  return p[:,:phase_count].clamp(0,1)[:,:,halo:-halo,halo:-halo]
with torch.inference_mode():target=phases(model,encoded[0]).float()
def objective():
 errors=[]
 with torch.inference_mode():
  ferr=float((phases(model,encoded[0]).float()-target).abs().mean()*255)
  for mode,q in enumerate(qmodels):
   quantize(q,model,mode);error=(phases(q,encoded[mode+1]).float()-target)*255;pixel=error.abs().mean();pair=error.reshape(8,2,*error.shape[1:]);temporal=(pair[:,1]-pair[:,0]).abs().mean();errors.append(float(pixel+.5*temporal))
 return max(errors)+.25*sum(errors)+ferr,{'float_mae_gray':ferr,'weight8_per_output_loss':errors[0],'weight8_per_tensor_loss':errors[1]}
def apply(edge,channel,exponent):
 _,first,last,ref=edge;scale=2.**exponent
 with torch.no_grad():
  first.weight[channel].mul_(scale);first.bias[channel].mul_(scale);last.weight[:,channel].div_(scale)
  if ref is not None:ref.weight[channel].mul_(scale);ref.bias[channel].mul_(scale)
def valid():
 return all(torch.isfinite(v).all() and torch.equal(v,v.half().float()) for v in model.parameters())
original=copy.deepcopy(model);initial_score,initial=objective();best=initial_score;changes=[];logs=[]
for sweep in range(3):
 count=0
 for edge in edges:
  for channel in range(edge[1].out_channels):
   chosen=None;chosen_score=best
   for exponent in (-1,1):
    apply(edge,channel,exponent)
    if valid():
     score,metrics=objective()
     if score<chosen_score-1e-5:chosen=(exponent,metrics);chosen_score=score
    apply(edge,channel,-exponent)
   if chosen:
    apply(edge,channel,chosen[0]);best=chosen_score;count+=1;changes.append({'sweep':sweep,'edge':edge[0],'channel':channel,'exponent':chosen[0],'score':best})
 logs.append({'sweep':sweep,'accepted':count,'score':best});print('SEARCH',a.scene,a.case,logs[-1],flush=True)
 if count==0:break
final_score,final=objective();dmodel=copy.deepcopy(model).double();dold=copy.deepcopy(original).double();dcontext=context[:2].double();dx=x[:2].double();dg=grid[:2].double()
with torch.inference_mode():
 owner=dold if quarter else dold.core.model;ref=owner.reference if quarter else owner.global_reference;enc=ref.encode(dcontext);enc=enc+enc.mean((-2,-1),keepdim=True);error=float((phases(dmodel,enc,dx,dg,False)-phases(dold,enc,dx,dg,False)).abs().max())
assert error<1e-9 and valid()
name=f'{a.scene}_cal_{a.case}';torch.save({'model':{n:v.detach().cpu() for n,v in model.state_dict().items()},'source_case':a.case,'GT_used':False,'test_used':False},out/(name+'.pt'));(out/(name+'.json')).write_text(json.dumps({'scene':a.scene,'source_case':a.case,'seed':2799,'GT_used':False,'test_used_for_search':False,'calibration_split':'train, last four records excluded','samples':samples,'initial':initial,'final':final,'changes':changes,'logs':logs,'float64_phase_max_abs':error,'half_coefficients_exact':True,'new_inference_operators':0,'SDK_bias_activation_calibration_not_simulated':True},indent=2));print('FINISHED',name,initial,final,flush=True)
