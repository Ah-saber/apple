import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
from run_compact import GROUPS,sha,score
from baseline_contract import load_group,independent_control
from motion_residual import MotionResidual


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--kind',choices=['half','quarter'],required=True);p.add_argument('--steps',type=int,default=8000);p.add_argument('--probe',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    compact,base,config,_,base_path,df,box_for=load_group(a.group);control=independent_control(a.group,compact)
    original=(base if a.kind=='half' else compact).cuda().float().eval()
    probe=json.loads((a.probe or a.root/'adaptive_probe_correct/adaptive_probe.json').read_text())['groups'][a.group]['kinds'][a.kind]
    model=MotionResidual(original,a.kind,probe['chosen_threshold']).cuda().float().eval()
    cache=ROOT/'runs'/('SS928-FIVE-SCENE-'+GROUPS[a.group]['cache']+'-20260929')/'train_cache.pt'
    data=torch.load(cache,map_location='cpu',weights_only=False)['values'];length=len(data['stack']);factor=model.factor
    with torch.inference_mode():
        records=[]
        for start in range(0,length,8):
            x=data['stack'][start:start+8].flatten(0,1).cuda().float();c=data['context'][start:start+8].flatten(0,1).cuda().float();box=data['box'][start:start+8].flatten(0,1).cuda().float()
            b,features,gate=model.extract(x,c,box);expected=original.native(x,c,box);actual=F.pixel_shuffle(b,factor)
            assert float((actual-expected).abs().max())*255<.01
            records.append((b.cpu(),features.cpu(),gate.cpu()))
        cached={name:torch.cat([v[i] for v in records]).reshape(length,2,*records[0][i].shape[1:]) for i,name in enumerate(['base','features','gate'])}
    cached['gt']=F.pixel_unshuffle(data['gt'].flatten(0,1).float(),factor).reshape(length,2,factor**2,128//factor,128//factor)
    delta=(data['gt'][:,1]-data['gt'][:,0]).float()*255;centered=delta-delta.flatten(1).median(1).values[:,None,None,None]
    smooth=F.avg_pool2d(centered,3,1,1);absolute=F.avg_pool2d(centered.abs(),3,1,1)
    candidates=(smooth.abs()>=3)&(smooth.abs()>=.6*absolute);masks=[]
    for mask in candidates[:,0].numpy():
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8);kept=np.zeros_like(mask)
        for label in range(1,len(stats)):
            xx,yy,ww,hh,area=stats[label]
            if 4<=area<=600 and ww<=60 and hh<=35:kept|=labels==label
        masks.append(kept)
    moving=F.max_pool2d(torch.from_numpy(np.stack(masks)[:,None]).float(),7,1,3)
    cached['mask']=F.pixel_unshuffle(moving,factor).unsqueeze(1).expand(-1,2,-1,-1,-1).clone()
    torch.save(cached,a.out/'train_features.pt')
    for key in cached:cached[key]=cached[key].cuda()
    probabilities=moving.flatten(1).sum(1).numpy();probabilities=.5/length+.5*(probabilities+.001)/(probabilities+.001).sum()
    rng=np.random.default_rng(930);opt=torch.optim.AdamW(model.residual.parameters(),lr=.002,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-5);logs=[];best=-np.inf;val=df(config,'val');started=time.perf_counter()
    margin=12//factor
    def validate(step):
        nonlocal best
        result=score(model,val,config,GROUPS[a.group]['scenes'],box_for);mean=np.mean([v['psnr_db'] for v in result['summary'].values()])
        state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'group':a.group,'kind':a.kind,'threshold':probe['chosen_threshold'],
               'saved_architecture':{'late_reference':model.q.late_reference} if a.kind=='quarter' else {'source_kind':'half'},
               'independent_baseline_control':control,'step':step,'validation':result,'original_denoiser_frozen':True,'test_used_for_selection':False}
        torch.save(state,a.out/f'step_{step:06d}.pt')
        if mean>best:best=mean;torch.save(state,a.out/'best.pt')
        return result['summary']
    logs.append({'step':0,'validation':validate(0)})
    for step in range(1,a.steps+1):
        ids=rng.choice(length,8,p=probabilities);model.residual.train()
        features=cached['features'][ids].flatten(0,1);b=cached['base'][ids].flatten(0,1);g=cached['gate'][ids].flatten(0,1)
        target=cached['gt'][ids].flatten(0,1);mask=cached['mask'][ids].flatten(0,1)
        pred=(b+g*model.residual(features)).clamp(0,1);e=(pred-target)[:,:,margin:-margin,margin:-margin]*255
        mask=mask[:,:,margin:-margin,margin:-margin];static=1-mask
        drift=(pred-b.clamp(0,1))[:,:,margin:-margin,margin:-margin]*255
        motion=(e.abs()*mask).sum()/mask.sum().clamp_min(1)
        anchor=(drift.abs()*static).sum()/static.sum().clamp_min(1)
        pair=e.reshape(8,2,*e.shape[1:]);temporal=(pair[:,1]-pair[:,0]).abs();paired_mask=mask.reshape(8,2,*mask.shape[1:])[:,1]
        temporal_motion=(temporal*paired_mask).sum()/paired_mask.sum().clamp_min(1)
        phase_means=e.mean((-2,-1));phase_bias=(phase_means-phase_means.mean(1,keepdim=True)).abs().mean()
        loss=e.abs().mean()+3*motion+2*temporal_motion+2*anchor+.1*phase_bias
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.residual.parameters(),5);opt.step();scheduler.step()
        if step==1 or step%1000==0:
            record={'step':step,'loss':float(loss),'motion_mae':float(motion),'static_anchor':float(anchor),'elapsed_seconds':time.perf_counter()-started}
            if step%2000==0:record['validation']=validate(step)
            logs.append(record)
            (a.out/'training.json').write_text(json.dumps({'group':a.group,'kind':a.kind,'steps':a.steps,'seed':930,'precision':'FP32; TF32 disabled',
                'cache':str(cache),'cache_sha256':sha(cache),'features_sha256':sha(a.out/'train_features.pt'),
                'original_denoiser_frozen':True,'zero_initial_residual_control':'all train cache predicted phases agree with original native within .01 gray',
                'independent_baseline_control':control,'target_mask':'training GT signed coherent component change; same limits as train_fp32',
                'logs':logs,'packaged':False,'pushed':False},indent=2))
            print('RESIDUAL_TRAIN',a.group,a.kind,json.dumps(record),flush=True)
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    test=score(model,df(config,'test'),config,GROUPS[a.group]['scenes'],box_for)
    (a.out/'test_12frames.json').write_text(json.dumps(test,indent=2));print('RESIDUAL_TEST',a.group,a.kind,json.dumps(test['summary']),flush=True)


if __name__=='__main__':main()
