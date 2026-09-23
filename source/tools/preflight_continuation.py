"""Verify one real optimizer update is continuous across the explicit schedule extension."""
import argparse,fcntl,json,os,sys
from pathlib import Path
import torch
C=Path(__file__).resolve().parents[1];sys.path.insert(0,str(C/'src'))
from ir_sr.training import dataset_for_config,epoch_loader,optimizer_for,restore_checkpoint,learning_rate,sha,atomic_json
from ir_sr.auxiliary import training_model,loss_terms
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
R=Path('/data/zhangbenzhuang/huawei_sr/runs');oldrun=R/'SS928-STUDENT-V2-20260923-S03-LIGHT_MEDIUM-8K';checkpoint=oldrun/'checkpoints/step_000008000.pt';old=json.loads((oldrun/'config.json').read_text());new=json.loads((C/'configs/train/student_s03_light_medium_200k.json').read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==old['gpu_uuid'];lease=(R/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
source=torch.load(checkpoint,map_location='cpu',weights_only=False);progress=source['progress'];dataset=dataset_for_config(new,'train',True);loader,_=epoch_loader(dataset,new,progress['epoch'],progress['next_batch']);iterator=iter(loader);batch=next(iterator)
v={k:batch[k].cuda() for k in ['raw','gt','middle_raw','raw_loss_multiplier','context','context_box']};states=[];losses=[]
for config,extension in [(old,False),(new,True)]:
 model=training_model(config).cuda();optimizer=optimizer_for(model,config);pos,best=restore_checkpoint(checkpoint,model,optimizer,config,extension)
 assert pos==progress and best==source['best_val_macro_psnr']
 for k,tensor in model.state_dict().items():torch.testing.assert_close(tensor.cpu(),source['model'][k],atol=0,rtol=0)
 for group in optimizer.param_groups:group['lr']=learning_rate(pos['step']+1,config)
 optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=loss_terms(model,v['raw'],v['gt'],v['middle_raw'],.1,v['raw_loss_multiplier'],v['context'],v['context_box'])
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
 states.append({k:x.detach().cpu().clone() for k,x in model.state_dict().items()});del model,optimizer,loss;torch.cuda.empty_cache()
error=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]);assert error<1e-7 and abs(losses[0]-losses[1])<1e-7
atomic_json(a.output/'report.json',dict(status='passed',source_checkpoint=str(checkpoint),sha256=sha(checkpoint),progress=progress,first_continuation_step=pos['step']+1,lr=learning_rate(pos['step']+1,new),losses=losses,post_update_max_parameter_abs_difference=error,samples=batch['sample_id'],peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
print(json.dumps(dict(status='passed',step=pos['step']+1,losses=losses,max_difference=error)),flush=True)
