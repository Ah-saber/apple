"""Constrained discrete output-weight quantization search on train ROI features."""
import argparse,fcntl,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-STRUCTURAL-20260927'
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from structural_candidates import load_structural
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(32,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2759);rng=np.random.default_rng(2759);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
teacher=load_structural(a.scene,'phase3_preserve_nearest',out);student=torch.nn.Conv2d(16,9,1).cuda().to(memory_format=torch.channels_last)
with torch.no_grad():student.weight.copy_(teacher.output.conv.weight.float().sum((-2,-1),keepdim=True));student.bias.copy_(teacher.output.conv.bias.float())
axis=torch.tensor([[2/3,1/3,0],[0,1/3,2/3]],device='cuda');mapping=torch.kron(axis,axis)[:,:,None,None]
X=[];Y=[];metadata=[];size=64
with torch.no_grad():
 for row in rows:
  features=[];targets=[];cy,cx,ch,cw=row['train_roi_tlhw'];assert cy%2==0 and cx%2==0
  for frame in [max(0,row['frame_id']-1),row['frame_id']]:
   rr=dict(row,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();c,m=teacher.core,teacher.core.model
   v=m.body(m.head(c.half_input(teacher.front(x))));ref=m.global_reference.encode(c.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));v=m.tail(v+F.interpolate(ref,size=v.shape[-2:],mode='bilinear',align_corners=False));features.append(v);targets.append(teacher.output.conv(v))
  for _ in range(6):
   y=cy//2+int(rng.integers(0,ch//2-size+1));x=cx//2+int(rng.integers(0,cw//2-size+1));X.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in features]).cpu());Y.append(torch.cat([v[:,:,y:y+size,x:x+size] for v in targets]).cpu());metadata.append({'sample_id':row['sample_id'],'frames':[max(0,row['frame_id']-1),row['frame_id']],'raw_crop':[y*2,x*2,size*2,size*2],'train_roi':row['train_roi_tlhw']})
  print('CACHE',a.scene,row['frame_id'],flush=True)
initial=teacher.output.conv.weight.detach().cpu().double().numpy().reshape(9,-1);bias=teacher.output.conv.bias.detach().cpu().clone();del teacher;torch.cuda.empty_cache()
covariance=np.zeros((144,144),np.float64);count=0
for features in X:
    v=F.unfold(features.double(),3).permute(0,2,1).numpy()
    flat=v.reshape(-1,144);change=v[1]-v[0]
    covariance+=flat.T@flat+2*(change.T@change);count+=len(flat)
covariance/=count
axis=np.array([[2/3,1/3,0],[0,1/3,2/3]],np.float64);A=np.kron(axis,axis);N=np.eye(9)-A.T@np.linalg.inv(A@A.T)@A
maximum=np.max(np.abs(initial),axis=1);scales=np.stack([maximum/127,np.full(9,maximum.max()/127)])
# Protect max coefficients, so both assumed quantizer scales remain fixed.
protected=np.any(np.isclose(np.abs(initial),maximum[:,None],atol=0,rtol=0),axis=0)
weights=initial.copy();quantized=np.clip(np.rint(weights[None]/scales[:,:,None]),-127,127)*scales[:,:,None]
error=np.einsum('ab,mbc->mac',A,quantized-initial);product=error@covariance
loss=float(np.sum(product*error)/8);initial_loss=loss;logs=[];started=time.perf_counter()
for epoch in range(10):
    accepted=0
    for column in rng.permutation(144):
        if protected[column]:continue
        best=None;best_change=0.
        for direction in range(9):
            for step in [-2,-1,-.5,-.25,.25,.5,1,2]:
                proposal=weights[:,column]+N[:,direction]*np.median(scales)*step
                if np.any(np.abs(proposal)>maximum):continue
                q=np.clip(np.rint(proposal[None]/scales),-127,127)*scales
                delta=np.einsum('ab,mb->ma',A,q-quantized[:,:,column])
                change=float((2*np.sum(delta*product[:,:,column])+covariance[column,column]*np.sum(delta*delta))/8)
                if change<best_change-1e-16:best_change=change;best=(proposal,q,delta)
        if best is not None:
            proposal,q,delta=best;weights[:,column]=proposal;quantized[:,:,column]=q;error[:,:,column]+=delta;product+=delta[:,:,None]*covariance[column][None,None,:];loss+=best_change;accepted+=1
    item={'epoch':epoch+1,'accepted':accepted,'predicted_native_mse_gray':loss*255**2,'elapsed_s':time.perf_counter()-started};logs.append(item);print('SEARCH',a.scene,item,flush=True)
    if not accepted:break
post_store=weights.reshape(9,16,3,3).astype(np.float16)
proof={'pre_store_native_weight_change_max':float(np.abs(A@(weights-initial)).max()),'post_store_native_weight_change_max':float(np.abs(A@(post_store.astype(np.float64).reshape(9,-1)-initial)).max()),'max_coefficients_protected':bool(np.all(np.max(np.abs(weights),axis=1)==maximum))}
assert proof['pre_store_native_weight_change_max']<1e-12
path=out/f'{a.scene}_output_quantsearch.pt'
if path.exists():raise FileExistsError(path)
torch.save({'format':'nine_phase_output_quantsearch_v1','scene':a.scene,'projection':{'weight':torch.from_numpy(post_store),'bias':bias},'proof':proof,'GT_used':False},path)
(out/f'{a.scene}_output_quantsearch_training.json').write_text(json.dumps({'scene':a.scene,'seed':2759,'train_samples':metadata,'GT_used':False,'test_used_for_training_or_checkpoint_selection':False,'method':'discrete nullspace search; float native means constrained; two symmetric127 weight8 assumptions; training feature and temporal covariance','initial_predicted_native_mse_gray':initial_loss*255**2,'final_predicted_native_mse_gray':loss*255**2,'proof':proof,'logs':logs,'SDK_verified':False,'activation16_simulated':False},indent=2));print('COMPLETE',a.scene,flush=True)
