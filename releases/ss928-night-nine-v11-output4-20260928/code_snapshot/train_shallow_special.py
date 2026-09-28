"""Distill one-layer special-night reference encoder on training contexts."""
import argparse, copy, fcntl, json, sys, time
from pathlib import Path
import numpy as np
import torch
from reference_candidates import ShallowEncoderReference, load_board

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--v09-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kernel',type=int,choices=[3,5],required=True)
    p.add_argument('--steps',type=int,default=8000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.v09_run.parent/'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,str(a.root/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src'))
        from ir_sr.training import dataset_for_config
        run=a.root/'runs/SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1'
        state=torch.load(run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
        ds=dataset_for_config(state['config'],'train')
        records=[r for r in ds.records if r['scene_id']=='night_special']
        assert len(records)>40
        train_records,held_records=records[:-8],records[-8:]
        contexts=[]
        for row in records:
            ctx,_=ds.context_for(row)
            contexts.append(ctx.numpy().astype(np.float32))
        values=torch.from_numpy(np.stack(contexts)).cuda()
        teacher=load_board('special','rows32',a.v09_run).core.model.global_reference.eval()
        teacher.requires_grad_(False)
        student=ShallowEncoderReference(copy.deepcopy(teacher),a.kernel).float().cuda()
        optimizer=torch.optim.AdamW(student.encoder.parameters(),lr=.001,weight_decay=1e-6)
        best=float('inf');best_step=0;logs=[];torch.manual_seed(928+a.kernel);began=time.perf_counter()
        for step in range(1,a.steps+1):
            ids=torch.randint(0,len(train_records),(12,),device='cuda')
            ctx=values.index_select(0,ids)
            ctx=ctx*(.8+.4*torch.rand(12,1,1,1,device='cuda'))+(torch.rand(12,1,1,1,device='cuda')-.5)*.3
            with torch.no_grad():target=teacher.encode(ctx.half()).float()
            output=student.encode(ctx);loss=(output-target).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%500==0 or step==1:
                with torch.no_grad():
                    held=values[len(train_records):]
                    diff=student.encode(held)-teacher.encode(held.half()).float()
                    rmse=float(diff.square().mean().sqrt());mae=float(diff.abs().mean())
                item={'step':step,'held_rmse':rmse,'held_mae':mae,'train_loss':float(loss)}
                logs.append(item);print(item,flush=True)
                if rmse<best:
                    best,best_step=rmse,step
                    ck={'encoder':{k:v.detach().cpu().half().clone() for k,v in student.encoder.state_dict().items()},'scene':'special','kernel':a.kernel,'step':step,'held_rmse':rmse,'test_used':False}
                    torch.save(ck,a.output/f'special_shallow{a.kernel}_reference.pt')
        report={'scene':'special','kernel':a.kernel,'best_step':best_step,'best_held_rmse':best,'steps':a.steps,'seconds':time.perf_counter()-began,'train_records':[{'sample_id':r['sample_id'],'frame_id':r['frame_id']} for r in train_records],'held_records':[{'sample_id':r['sample_id'],'frame_id':r['frame_id']} for r in held_records],'logs':logs,'test_used':False}
        (a.output/f'special_shallow{a.kernel}_training.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
