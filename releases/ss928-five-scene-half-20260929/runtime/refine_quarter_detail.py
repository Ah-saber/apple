"""Train a cheap half-grid detail correction on a frozen quarter-grid model."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DetailRefiner(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base
        self.scale=1.0
        self.first=nn.Conv2d(12,12,3,padding=1)
        self.last=nn.Conv2d(12,4,3,padding=1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def native(self,stack,context,box):
        base=self.base.native(stack,context,box)
        current=stack[:,-1:].float()
        history=stack[:,:6].float().mean(1,keepdim=True)
        signal=torch.cat((F.pixel_unshuffle(base.float(),2),
            F.pixel_unshuffle(current-base.float(),2),
            F.pixel_unshuffle(current-history,2)),1)
        correction=F.pixel_shuffle(self.last(F.relu(self.first(signal.to(
            dtype=self.first.weight.dtype,memory_format=torch.channels_last)))),2)
        return base.float()+correction.float()*self.scale

    def forward(self,stack,context,box):
        return F.interpolate(self.native(stack,context,box).clamp(0,1)*255,
                             scale_factor=3,mode='nearest')


def load_base(code,night_runtime,config_checkpoint,base_checkpoint,
              reference_checkpoint=None):
    sys.path[:0]=[str(code/'src'),str(night_runtime),str(Path(__file__).parent)]
    from ir_sr.model import inference_model
    from half_student import make_student
    from train_quarter_student import BoxQuarter
    initial=torch.load(config_checkpoint,map_location='cpu',weights_only=False)
    reference_state=(torch.load(reference_checkpoint,map_location='cpu',
                     weights_only=False) if reference_checkpoint else initial)
    source=inference_model(reference_state['config'],reference_state['model'])
    ck=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
    base=BoxQuarter(make_student(copy.deepcopy(source.global_reference),ck),
        current_only=ck.get('current_only',False))
    base.load_state_dict(ck['model'],strict=True)
    return base,initial['config'],ck


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--night-runtime',type=Path,required=True)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--train-cache',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--motion-weight',type=float,default=.5)
    p.add_argument('--teacher-weight',type=float,default=.5)
    p.add_argument('--weak-weight',type=float,default=.3)
    p.add_argument('--static-weight',type=float,default=.3)
    p.add_argument('--pair-weight',type=float,default=.3)
    p.add_argument('--seed',type=int,default=2930)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base,config,base_ck=load_base(a.code,a.night_runtime,
        a.config_checkpoint,a.base_checkpoint,a.reference_checkpoint)
    from train_quarter_student import validation
    from ir_sr.training import dataset_for_config
    base.requires_grad_(False)
    model=DetailRefiner(base).cuda().to(memory_format=torch.channels_last)
    saved=torch.load(a.train_cache,map_location='cpu',weights_only=False)
    data=saved['values']
    metadata=saved['metadata']
    ids_pool=np.asarray([i for i,v in enumerate(metadata)
                         if v['scene'] in a.scenes],dtype=int)
    if len(ids_pool)<20:
        raise ValueError(f'Insufficient cached training pairs: {len(ids_pool)}')
    val=dataset_for_config(config,'val')
    opt=torch.optim.AdamW(list(model.first.parameters())+
                          list(model.last.parameters()),lr=a.lr,weight_decay=1e-6)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=2e-6)
    rng=np.random.default_rng(a.seed)
    logs=[]
    best=-float('inf')
    start=time.perf_counter()
    edge=slice(12,-12)
    for step in range(1,a.steps+1):
        ids=rng.choice(ids_pool,size=4,replace=True)
        x=data['stack'][ids].flatten(0,1).cuda().float()
        c=data['context'][ids].flatten(0,1).cuda().float()
        b=data['box'][ids].flatten(0,1).cuda().float()
        gt=data['gt'][ids].flatten(0,1).cuda().float()
        teacher=data['teacher'][ids].flatten(0,1).cuda().float()
        model.train()
        model.base.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.native(x,c,b).float().clamp(0,1)
            err=(pred-gt)*255
            imitation=((pred-teacher)*255)[:,:,edge,edge].abs().mean()
            pixel=err[:,:,edge,edge].abs().mean()
            pair=err.reshape(4,2,1,128,128)
            pair_error=(pair[:,1]-pair[:,0])[:,:,edge,edge].abs()
            gt_delta=(gt[1::2]-gt[::2]).abs()[:,:,edge,edge]*255
            motion=gt_delta>=3
            weak=(gt_delta>=1)&(gt_delta<3)
            motion_error=(pair_error*motion).sum()/motion.sum().clamp_min(1)
            weak_error=(pair_error*weak).sum()/weak.sum().clamp_min(1)
            static=F.avg_pool2d(err[:,:,edge,edge],8,8).abs().mean()
            loss=(pixel+a.teacher_weight*imitation+
                  a.pair_weight*pair_error.mean()+
                  a.motion_weight*motion_error+
                  a.weak_weight*weak_error+a.static_weight*static)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.first.parameters())+
                                 list(model.last.parameters()),5)
        opt.step()
        schedule.step()
        if step==1 or step%1000==0:
            row={'step':step,'loss':float(loss.detach()),
                 'pixel_gray':float(pixel.detach()),
                 'motion_error_gray':float(motion_error.detach()),
                 'elapsed_s':time.perf_counter()-start}
            if step%2000==0:
                row['val_psnr_db']=validation(model,val,config,a.scenes)
                row['val_macro_psnr_db']=float(np.mean(list(row['val_psnr_db'].values())))
                state={'format':'quarter_detail_refiner_v1','step':step,
                       'scenes':a.scenes,'base_checkpoint':str(a.base_checkpoint),
                       'config_checkpoint':str(a.config_checkpoint),
                       'reference_checkpoint':str(a.reference_checkpoint),
                       'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                       'val_psnr_db':row['val_psnr_db'],
                       'test_used_for_selection':False}
                torch.save(state,a.out/f'step_{step:06d}.pt')
                if row['val_macro_psnr_db']>best:
                    best=row['val_macro_psnr_db']
                    torch.save(state,a.out/'best.pt')
            logs.append(row)
            print('TRAIN',json.dumps(row),flush=True)
            (a.out/'training.json').write_text(json.dumps({
                'base_checkpoint':str(a.base_checkpoint),
                'scenes':a.scenes,'train_pairs':len(ids_pool),
                'motion_weight':a.motion_weight,
                'teacher_weight':a.teacher_weight,
                'weak_weight':a.weak_weight,
                'static_weight':a.static_weight,
                'pair_weight':a.pair_weight,
                'best_val_macro_psnr_db':best,'logs':logs},indent=2))


if __name__=='__main__':main()
