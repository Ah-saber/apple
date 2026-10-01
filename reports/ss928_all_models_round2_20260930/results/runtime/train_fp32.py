import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.nn import functional as F

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
from run_compact import setup,GROUPS,sha,score
from short_history import ShortHistory
from baseline_contract import load_group,independent_control


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--count',type=int,required=True)
    p.add_argument('--steps',type=int,default=16000);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--kind',choices=['half','quarter']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    compact,base,config,teacher,base_path,df,box_for=load_group(a.group)
    baseline_control=independent_control(a.group,compact)
    kind=a.kind or ('half' if a.group=='day' else 'quarter')
    path=base_path if kind=='half' else V13/(a.group+'_temporal')/'best.pt'
    if kind=='quarter':compact.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    source=(base if kind=='half' else compact).cuda().float().eval()
    model=ShortHistory(source,a.count,kind,current_bias=0.).cuda().to(memory_format=torch.channels_last)
    if a.resume:model.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=False)['model'])
    cache=ROOT/'runs'/('SS928-FIVE-SCENE-'+GROUPS[a.group]['cache']+'-20260929')/'train_cache.pt'
    saved=torch.load(cache,map_location='cpu',weights_only=False);data=saved['values'];metadata=saved['metadata']
    assert all(r['scene'] in GROUPS[a.group]['scenes'] for r in metadata)
    controls=[]
    with torch.inference_mode():
        x=data['stack'][:2].flatten(0,1).cuda().float();x=x[:,-1:].expand(-1,9,-1,-1)
        c=data['context'][:2].flatten(0,1).cuda().float();box=data['box'][:2].flatten(0,1).cuda().float()
        err=(model.native(x,c,box)-source.native(x,c,box)).abs()*255
        controls={'mae_gray':float(err.mean()),'max_gray':float(err.max())}
        if not a.resume:assert controls['max_gray']<.01,controls
        frozen_targets=[]
        for first in range(0,len(metadata),16):
            bx=data['stack'][first:first+16].flatten(0,1).cuda().float()
            bc=data['context'][first:first+16].flatten(0,1).cuda().float()
            bb=data['box'][first:first+16].flatten(0,1).cuda().float()
            frozen_targets.append(source.native(bx,bc,bb).clamp(0,1).cpu())
        frozen_targets=torch.cat(frozen_targets).reshape(len(metadata),2,1,128,128)
        torch.save(frozen_targets,a.out/'source_train_predictions.pt')
    model.reference.encoder.requires_grad_(False)
    if hasattr(model.reference,'pyramid'):model.reference.pyramid.requires_grad_(False)
    difference=(data['gt'][:,1]-data['gt'][:,0])*255
    centered=difference-difference.flatten(1).median(1).values[:,None,None,None]
    coherent=F.avg_pool2d(centered,3,1,1)
    mean_absolute=F.avg_pool2d(centered.abs(),3,1,1)
    candidates=(coherent.abs()>=3)&(coherent.abs()>=.6*mean_absolute)
    masks=[]
    for candidate in candidates[:,0].numpy():
        n,components,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
        kept=np.zeros_like(candidate)
        for label in range(1,n):
            xx,yy,ww,hh,area=stats[label]
            if 4<=area<=600 and ww<=60 and hh<=35:kept|=components==label
        masks.append(kept)
    target_mask=F.max_pool2d(torch.from_numpy(np.stack(masks)[:,None]).float(),7,1,3)[:,:,12:-12,12:-12]
    torch.save(target_mask,a.out/'training_target_masks.pt')
    strengths=target_mask.flatten(1).sum(1).numpy();probs=.5/len(strengths)+.5*(strengths+.001)/(strengths+.001).sum()
    rng=np.random.default_rng(930);opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=5e-5,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=2e-6)
    val=df(config,'val');logs=[];best=-np.inf;start=time.perf_counter()
    def validate(step):
        nonlocal best
        scores=score(model,val,config,GROUPS[a.group]['scenes'],box_for)
        value=float(np.mean([r['psnr_db'] for r in scores['summary'].values()]))
        ck={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'count':a.count,'kind':kind,'group':a.group,
            'saved_architecture':{'late_reference':source.late_reference} if kind=='quarter' else {'source_kind':'half'},
            'independent_baseline_control':baseline_control,
            'source':str(path),'source_sha256':sha(path),'step':step,'validation':scores,'test_used_for_selection':False}
        if value>best:best=value;torch.save(ck,a.out/'best.pt')
        torch.save(ck,a.out/f'step_{step:06d}.pt')
        return scores['summary']
    logs.append({'step':0,'validation':validate(0)})
    for step in range(1,a.steps+1):
        ids=rng.choice(len(metadata),4,p=probs)
        x=data['stack'][ids].flatten(0,1).cuda().float();c=data['context'][ids].flatten(0,1).cuda().float()
        box=data['box'][ids].flatten(0,1).cuda().float();gt=data['gt'][ids].flatten(0,1).cuda().float()
        teacher_target=frozen_targets[ids].flatten(0,1).cuda().float();model.train()
        with torch.autocast('cuda',enabled=False):
            pred=model.native(x,c,box).float().clamp(0,1);e=(pred-gt)[:,:,12:-12,12:-12]*255
            delta=(gt[1::2]-gt[::2])[:,:,12:-12,12:-12]*255
            # Dilate the paired GT motion regions to cover old and new target neighborhoods.
            mask=target_mask[ids].cuda()
            weight=mask.repeat_interleave(2,0)
            pair=e.reshape(4,2,1,104,104);change_error=(pair[:,1]-pair[:,0]).abs()
            moving=(e.abs()*weight).sum()/weight.sum().clamp_min(1)
            temporal=(change_error*mask).sum()/mask.sum().clamp_min(1)
            weak=((delta.abs()>=1)&(delta.abs()<3)).float()
            weak_loss=(change_error*weak).sum()/weak.sum().clamp_min(1)
            static=(1-weight)
            imitate=((pred-teacher_target).abs()[:,:,12:-12,12:-12]*static).sum()/static.sum().clamp_min(1)*255
            gy=((pred[:,:,1:]-pred[:,:,:-1])-(gt[:,:,1:]-gt[:,:,:-1])).abs()[:,:,12:-12,12:-12].mean()*255
            gx=((pred[:,:,:,1:]-pred[:,:,:,:-1])-(gt[:,:,:,1:]-gt[:,:,:,:-1])).abs()[:,:,12:-12,12:-12].mean()*255
            loss=2*e.abs().mean()+moving+temporal+.5*weak_loss+.5*change_error.mean()+2*imitate+.15*(gx+gy)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();scheduler.step()
        if step==1 or step%1000==0:
            record={'step':step,'loss':float(loss),'mae':float(e.abs().mean()),'elapsed_seconds':time.perf_counter()-start}
            if step%4000==0:record['validation']=validate(step)
            logs.append(record)
            (a.out/'training.json').write_text(json.dumps({'group':a.group,'kind':kind,'count':a.count,'steps':a.steps,'seed':930,
                'source':str(path),'source_sha256':sha(path),'cache':str(cache),'cache_sha256':sha(cache),
                'repeated_current_algebra_control':controls,'sampling_probabilities':probs.tolist(),'logs':logs,
                'independent_baseline_control':baseline_control,
                'resume':None if a.resume is None else {'path':str(a.resume),'sha256':sha(a.resume)},
                'static_anchor':'frozen source prediction on existing training cache; GT motion-dilated regions excluded',
                'static_anchor_sha256':sha(a.out/'source_train_predictions.pt'),
                'target_mask':'training GT pair scalar drift removed; coherent 3x3 signed change>=3; sign coherence .6; component area4..600 width<=60 height<=35; dilated3',
                'target_mask_sha256':sha(a.out/'training_target_masks.pt'),
                'training_precision':'FP32, TF32 disabled',
                'nine_frame_initial_current_bias':0.,
                'selection':'validation mean PSNR; save every 4000 steps for separate validation target review',
                'packaged':False,'pushed':False},indent=2))
            print('SHORT_TRAIN',a.group,a.count,json.dumps(record),flush=True)
    ck=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    result=score(model,df(config,'test'),config,GROUPS[a.group]['scenes'],box_for)
    (a.out/'test_12frames.json').write_text(json.dumps(result,indent=2));print('SHORT_TEST',a.group,a.count,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
