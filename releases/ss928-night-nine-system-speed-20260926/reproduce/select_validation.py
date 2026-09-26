"""Select front/reference checkpoints using validation ROIs and frozen-model targets."""
import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from load_packed_model import load_packed_model;from system_variants import QuarterFront,CompactReference,CompactBody
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'val');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,min(8,len(records)),dtype=int)]
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if not special else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{a.scene}_selection.json').read_text())['selected_step'];base=load_packed_model(run/f'{a.scene}_packed_front_{step:06d}.pt',root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt');cache=[];meta=[];bodycache=[]
with torch.inference_mode():
 for row in rows:
  roi=row['eval_crop_tlhw'];y,x,h,w=roi;h=h//4*4;w=w//4*4;inputs=[];targets=[];contexts=[];references=[]
  # Full-frame teacher inference, validation ROI only for comparison.
  for frame in (max(0,row['frame_id']-1),row['frame_id']):
   rr=dict(row,frame_id=frame);stack=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();target=base.front(stack);head=base.core.model.head(base.core.half_input(target));body_target=base.core.model.body(head);bodycache.append((head.cpu(),body_target.cpu(),(slice(y//2,(y+h)//2),slice(x//2,(x+w)//2))));ref=base.core.model.global_reference;feat=ref.encode(base.core.half_input(ctx));projected=ref.project(feat+feat.mean((-2,-1),keepdim=True));inputs.append(stack.cpu());targets.append(target.cpu());contexts.append(ctx.cpu());references.append(projected.float().cpu())
  cache.append((inputs,targets,contexts,references,(slice(y//2,(y+h)//2),slice(x//2,(x+w)//2))));meta.append({'sample_id':row['sample_id'],'frame_pair':[max(0,row['frame_id']-1),row['frame_id']],'validation_roi':roi})
 fronts={};refs={}
 for path in sorted(out.glob(f'{a.scene}_quarter*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=True);front=QuarterFront(ck['width'],ck['blocks']);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last);absolute=[];temporal=[]
  for inputs,targets,_,_,roi in cache:
   errs=[]
   for stack,target in zip(inputs,targets):
    e=(front(stack.cuda())-target.cuda())[0,:,roi[0],roi[1]];absolute.append(float(e.abs().mean()));errs.append(e)
   temporal.append(float((errs[1]-errs[0]).abs().mean()))
  record={'mae_normalized_raw':float(np.mean(absolute)),'temporal_pair_error_normalized_raw':float(np.mean(temporal))};record['score']=record['mae_normalized_raw']+.75*record['temporal_pair_error_normalized_raw'];fronts[path.name]=record;print('FRONT',a.scene,path.name,record,flush=True);del front
 for path in sorted(out.glob(f'{a.scene}_reference*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=True);reference=CompactReference(ck['width']);reference.load_state_dict(ck['reference']);reference=reference.cuda().half().to(memory_format=torch.channels_last);absolute=[];temporal=[]
  for _,_,contexts,targets,_ in cache:
   errs=[]
   for context,target in zip(contexts,targets):e=reference(context.cuda()).float()-target.cuda();absolute.append(float(e.abs().mean()));errs.append(e)
   temporal.append(float((errs[1]-errs[0]).abs().mean()))
  record={'feature_mae':float(np.mean(absolute)),'temporal_pair_feature_error':float(np.mean(temporal))};record['score']=record['feature_mae']+record['temporal_pair_feature_error'];refs[path.name]=record;print('REFERENCE',a.scene,path.name,record,flush=True);del reference
 bodies={}
 for path in sorted(out.glob(f'{a.scene}_body*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=True);body=CompactBody(ck['blocks']);body.load_state_dict(ck['body']);body=body.cuda().half().to(memory_format=torch.channels_last);errors=[]
  for head,target,roi in bodycache:
   e=(body(head.cuda())-target.cuda())[0,:,roi[0],roi[1]];errors.append(float(e.abs().mean()))
  bodies[path.name]={'feature_mae':float(np.mean(errors))};print('BODY',a.scene,path.name,bodies[path.name],flush=True);del body
 result={'scene':a.scene,'split':'val','teacher':'frozen v0.6 source','validation_samples':meta,'front_candidates':fronts,'reference_candidates':refs,'body_candidates':bodies,'selected_body':min(bodies,key=lambda k:bodies[k]['feature_mae']),'selected_front':min(fronts,key=lambda k:fronts[k]['score']),'selected_reference':min(refs,key=lambda k:refs[k]['score']),'GT_used_for_selection':False,'full_output_quality_not_yet_verified':True};(out/f'{a.scene}_selection.json').write_text(json.dumps(result,indent=2));print('SELECTED',result['selected_front'],result['selected_reference'],flush=True)
