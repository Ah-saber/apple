"""Distill v0.8 multiscale reference on training input contexts only; no GT/test labels."""
import fcntl,json,sys,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
R=Path('/data/zhangbenzhuang/huawei_sr');RUN=R/'runs/SS928-BOARD-V09-20260928';sys.path.insert(0,str(Path(__file__).parent));from board_candidates import load_board,StudentReference
lease=(R/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.manual_seed(2810);np.random.seed(2810);random.seed(2810);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
results=[]
for scene in ('ordinary','special'):
 if scene=='ordinary' and all((RUN/f'ordinary_reference_student{d}.pt').exists() for d in (3,4)):
  print('REUSE_COMPLETED_ORDINARY_CHECKPOINTS',flush=True);continue
 special=scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(R/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
 state=torch.load(R/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+scene];assert len(records)>40;training=records[:-8];held=records[-8:];contexts=[];source=[]
 for row in training+held:
  ctx,_=ds.context_for(row);contexts.append(ctx.numpy().astype(np.float32));source.append({'sample_id':row['sample_id'],'frame_id':row['frame_id'],'split':'training' if len(source)<len(training) else 'train_heldout'})
 values=torch.from_numpy(np.stack(contexts)).cuda();train_count=len(training);teacher=load_board(scene,'v08_c11',RUN).core.model.global_reference.eval();teacher.requires_grad_(False);print('DATA',scene,train_count,len(held),flush=True)
 for depth in (3,4):
  model=StudentReference(teacher,depth).float().cuda();student=model.encoder;optimizer=torch.optim.AdamW(student.parameters(),lr=.001,weight_decay=1e-6);best=float('inf');best_step=0;logs=[];start=time.perf_counter();attempts=8000
  for step in range(1,attempts+1):
   rng=torch.randint(0,train_count,(12,),device='cuda');ctx=values.index_select(0,rng);gain=.8+.4*torch.rand(12,1,1,1,device='cuda');offset=(torch.rand(12,1,1,1,device='cuda')-.5)*.3;ctx=ctx*gain+offset
   with torch.no_grad():target=teacher.encode(ctx.half()).float()
   pred=student(ctx);den=target.detach().square().mean((0,2,3),keepdim=True).clamp_min(.005);loss=((pred-target).square()/den).mean()+.1*(pred-target).abs().mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);optimizer.step()
   for group in optimizer.param_groups:group['lr']=.0001+.0009*.5*(1+np.cos(np.pi*step/attempts))
   if step%250==0 or step==attempts:
    with torch.inference_mode():
     hold=values[train_count:];out=student(hold);t=teacher.encode(hold.half()).float();held_mae=float((out-t).abs().mean());held_max=float((out-t).abs().max());held_rmse=float((out-t).square().mean().sqrt())
    row={'step':step,'train_loss':float(loss),'held_mae':held_mae,'held_rmse':held_rmse,'held_max':held_max};logs.append(row);print('REF_TRAIN',scene,depth,row,flush=True)
    if held_rmse<best:
     best=held_rmse;best_step=step;torch.save({'student':{k:v.detach().cpu().clone() for k,v in student.state_dict().items()},'scene':scene,'depth':depth,'best_step':step,'held_rmse':held_rmse,'teacher_case':'v08_c11','teacher_multiscale_retained_as_source':True,'train_record_count':train_count,'held_record_count':len(held)},RUN/f'{scene}_reference_student{depth}.pt')
  results.append({'scene':scene,'depth':depth,'attempts':attempts,'best_step':best_step,'held_rmse':best,'seconds':time.perf_counter()-start,'train_records':[{'sample_id':x['sample_id'],'frame_id':x['frame_id']} for x in training],'held_records':[{'sample_id':x['sample_id'],'frame_id':x['frame_id']} for x in held],'GT_used':False,'test_inputs_used':False,'rows':logs})
  del model,student,optimizer;torch.cuda.empty_cache()
 del teacher,values;torch.cuda.empty_cache()
(RUN/'reference_training.json').write_text(json.dumps({'results':results,'SDK_verified':False},indent=2));print('REFERENCE_TRAIN_COMPLETE',flush=True)
