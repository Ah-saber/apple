"""Train a review-only small model; select on val and save independent controls."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import ROOT,setup,GROUPS,score,sha
from reliable_model import ReliableModel
from native_reliable_model import NativeReliableModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['light','heavy'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--width',type=int,default=8);p.add_argument('--resume',type=Path)
    p.add_argument('--native-gate',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    _,base,config,teacher,base_path,df,box_for=setup(a.group)
    kind='native' if a.native_gate else 'coarse'
    threshold=8. if a.native_gate else 2.
    cls=NativeReliableModel if a.native_gate else ReliableModel
    model=cls(base.quarter.reference,width=a.width,threshold_gray=threshold).cuda().to(memory_format=torch.channels_last)
    initial_control=None
    if a.resume:
        saved_initial=torch.load(a.resume,map_location='cpu',weights_only=False)
        old_cls=NativeReliableModel if saved_initial.get('kind')=='native' else ReliableModel
        old=old_cls(base.quarter.reference,width=saved_initial['width'],threshold_gray=saved_initial['threshold_gray']).cuda()
        old.load_state_dict(saved_initial['model'])
        n=saved_initial['width'];assert a.width>=n
        with torch.no_grad():
            model.reference.load_state_dict(old.reference.state_dict())
            model.front.weight[:n].copy_(old.front.weight);model.front.bias[:n].copy_(old.front.bias)
            model.body[0].weight[:n].zero_();model.body[0].weight[:n,:n].copy_(old.body[0].weight)
            model.body[0].bias[:n].copy_(old.body[0].bias)
            model.head.weight.zero_();model.head.weight[:,:n].copy_(old.head.weight);model.head.bias.copy_(old.head.bias)
            random=torch.rand(1,1,128,128,device='cuda').expand(1,9,128,128)
            context=torch.rand(1,old.reference.input_channels,64,64,device='cuda')
            box=torch.tensor([[0.,0.,1.,1.]],device='cuda')
            d=(model.native(random,context,box)-old.native(random,context,box)).abs()*255
            initial_control={'mae_gray':float(d.mean()),'max_gray':float(d.max())}
            assert initial_control['max_gray']<.01,initial_control
    model.reference.encoder.requires_grad_(False)
    if hasattr(model.reference,'pyramid'):model.reference.pyramid.requires_grad_(False)
    spec=GROUPS[a.group]
    cache=ROOT/'runs'/('SS928-FIVE-SCENE-'+spec['cache']+'-20260929')/'train_cache.pt'
    saved=torch.load(cache,map_location='cpu',weights_only=False);data=saved['values'];metadata=saved['metadata']
    assert all(r['scene'] in spec['scenes'] for r in metadata)
    # Mix uniform coverage and GT-motion oversampling within the existing train cache.
    motion=((data['gt'][:,1]-data['gt'][:,0]).abs()*255>=3).float().mean((1,2,3)).numpy()
    probabilities=.5/len(motion)+.5*(motion+.001)/(motion+.001).sum()
    rng=np.random.default_rng(930);logs=[];best=-np.inf;start=time.perf_counter()
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4 if a.resume else 2e-4,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=2e-6)
    val=df(config,'val')
    def save_state(step,scores):
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'width':a.width,'threshold_gray':threshold,'kind':kind,
            'reference_checkpoint':str(base_path),'reference_sha256':sha(base_path),'config':config,'group':a.group,
            'step':step,'val':scores,'test_used_for_selection':False},a.out/'best.pt')
    if a.resume:
        initial_scores=score(model,val,config,spec['scenes'],box_for)
        best=float(np.mean([r['psnr_db'] for r in initial_scores['summary'].values()]));save_state(0,initial_scores)
        logs.append({'step':0,'validation':initial_scores['summary'],'widening_control':initial_control})
    for step in range(1,a.steps+1):
        ids=rng.choice(len(metadata),4,p=probabilities)
        x=data['stack'][ids].flatten(0,1).cuda().float();c=data['context'][ids].flatten(0,1).cuda().float()
        box=data['box'][ids].flatten(0,1).cuda().float();gt=data['gt'][ids].flatten(0,1).cuda().float()
        teacher_target=data['teacher'][ids].flatten(0,1).cuda().float()
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.native(x,c,box).float().clamp(0,1)
            e=(pred-gt)[:,:,12:-12,12:-12]*255
            error=e.reshape(4,2,1,104,104)
            delta=(gt[1::2]-gt[::2])[:,:,12:-12,12:-12]*255
            moving=delta.abs()>=3;weak=(delta.abs()>=1)&(delta.abs()<3)
            temporal=(error[:,1]-error[:,0]).abs()
            motion_loss=(temporal*moving).sum()/moving.sum().clamp_min(1)
            weak_loss=(temporal*weak).sum()/weak.sum().clamp_min(1)
            grad_y=((pred[:,:,1:]-pred[:,:,:-1])-(gt[:,:,1:]-gt[:,:,:-1])).abs()[:,:,12:-12,12:-12].mean()*255
            grad_x=((pred[:,:,:,1:]-pred[:,:,:,:-1])-(gt[:,:,:,1:]-gt[:,:,:,:-1])).abs()[:,:,12:-12,12:-12].mean()*255
            imitate=(pred-teacher_target).abs()[:,:,12:-12,12:-12].mean()*255
            loss=e.abs().mean()+.1*imitate+temporal.mean()+3*motion_loss+weak_loss+.25*(grad_x+grad_y)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        opt.step();scheduler.step()
        if step==1 or step%1000==0:
            record={'step':step,'loss':float(loss),'mae_gray':float(e.abs().mean()),'elapsed_seconds':time.perf_counter()-start}
            if step%2000==0:
                scores=score(model,val,config,spec['scenes'],box_for)
                record['validation']=scores['summary'];metric=np.mean([r['psnr_db'] for r in scores['summary'].values()])
                if metric>best:
                    best=metric
                    save_state(step,scores)
            logs.append(record)
            (a.out/'training.json').write_text(json.dumps({'group':a.group,'steps':a.steps,'seed':930,'train_cache':str(cache),'cache_sha256':sha(cache),
                'motion_sampling_probabilities':probabilities.tolist(),'threshold_gray':threshold,'kind':kind,'width':a.width,'logs':logs,
                'resume':None if not a.resume else {'path':str(a.resume),'sha256':sha(a.resume)},'widening_control':initial_control,
                'packaged':False,'pushed':False},indent=2))
            print('TRAIN_RELIABLE',a.group,json.dumps(record),flush=True)
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    test=df(config,'test')
    result=score(model,test,config,spec['scenes'],box_for)
    (a.out/'test_12frames.json').write_text(json.dumps(result,indent=2))
    print('RELIABLE_TEST',a.group,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
