import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
from run_compact import setup,GROUPS,sha,score
from short_history import ShortHistory
from baseline_contract import load_group,independent_control


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--count',type=int,required=True)
    p.add_argument('--steps',type=int,default=16000);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    compact,base,config,teacher,base_path,df,box_for=load_group(a.group)
    baseline_control=independent_control(a.group,compact)
    kind='half' if a.group=='day' else 'quarter'
    path=base_path if kind=='half' else V13/(a.group+'_temporal')/'best.pt'
    if kind=='quarter':compact.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    source=(base if kind=='half' else compact).cuda().float().eval()
    model=ShortHistory(source,a.count,kind).cuda().to(memory_format=torch.channels_last)
    cache=ROOT/'runs'/('SS928-FIVE-SCENE-'+GROUPS[a.group]['cache']+'-20260929')/'train_cache.pt'
    saved=torch.load(cache,map_location='cpu',weights_only=False);data=saved['values'];metadata=saved['metadata']
    assert all(r['scene'] in GROUPS[a.group]['scenes'] for r in metadata)
    controls=[]
    with torch.inference_mode():
        x=data['stack'][:2].flatten(0,1).cuda().float();x=x[:,-1:].expand(-1,9,-1,-1)
        c=data['context'][:2].flatten(0,1).cuda().float();box=data['box'][:2].flatten(0,1).cuda().float()
        err=(model.native(x,c,box)-source.native(x,c,box)).abs()*255
        controls={'mae_gray':float(err.mean()),'max_gray':float(err.max())};assert controls['max_gray']<.01,controls
    model.reference.encoder.requires_grad_(False)
    if hasattr(model.reference,'pyramid'):model.reference.pyramid.requires_grad_(False)
    motion=(data['gt'][:,1]-data['gt'][:,0]).abs()*255
    strengths=motion.flatten(1).mean(1).numpy();probs=.5/len(strengths)+.5*(strengths+.001)/(strengths+.001).sum()
    rng=np.random.default_rng(930);opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=5e-5,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=2e-6)
    val=df(config,'val');logs=[];best=-np.inf;start=time.perf_counter()
    def validate(step):
        nonlocal best
        scores=score(model,val,config,GROUPS[a.group]['scenes'],box_for)
        value=float(np.mean([r['psnr_db'] for r in scores['summary'].values()]))
        ck={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'count':a.count,'kind':kind,'group':a.group,
            'source':str(path),'source_sha256':sha(path),'step':step,'validation':scores,'test_used_for_selection':False}
        if value>best:best=value;torch.save(ck,a.out/'best.pt')
        torch.save(ck,a.out/f'step_{step:06d}.pt')
        return scores['summary']
    logs.append({'step':0,'validation':validate(0)})
    for step in range(1,a.steps+1):
        ids=rng.choice(len(metadata),4,p=probs)
        x=data['stack'][ids].flatten(0,1).cuda().float();c=data['context'][ids].flatten(0,1).cuda().float()
        box=data['box'][ids].flatten(0,1).cuda().float();gt=data['gt'][ids].flatten(0,1).cuda().float()
        teacher_target=data['teacher'][ids].flatten(0,1).cuda().float();model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.native(x,c,box).float().clamp(0,1);e=(pred-gt)[:,:,12:-12,12:-12]*255
            delta=(gt[1::2]-gt[::2])[:,:,12:-12,12:-12]*255
            # Dilate the paired GT motion regions to cover old and new target neighborhoods.
            mask=F.max_pool2d((delta.abs()>=3).float(),7,1,3)
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
            loss=e.abs().mean()+2*moving+2*temporal+weak_loss+.5*change_error.mean()+.05*imitate+.15*(gx+gy)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();scheduler.step()
        if step==1 or step%1000==0:
            record={'step':step,'loss':float(loss),'mae':float(e.abs().mean()),'elapsed_seconds':time.perf_counter()-start}
            if step%4000==0:record['validation']=validate(step)
            logs.append(record)
            (a.out/'training.json').write_text(json.dumps({'group':a.group,'kind':kind,'count':a.count,'steps':a.steps,'seed':930,
                'source':str(path),'source_sha256':sha(path),'cache':str(cache),'cache_sha256':sha(cache),
                'repeated_current_algebra_control':controls,'sampling_probabilities':probs.tolist(),'logs':logs,
                'independent_baseline_control':baseline_control,
                'selection':'validation mean PSNR; save every 4000 steps for separate validation target review',
                'packaged':False,'pushed':False},indent=2))
            print('SHORT_TRAIN',a.group,a.count,json.dumps(record),flush=True)
    ck=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    result=score(model,df(config,'test'),config,GROUPS[a.group]['scenes'],box_for)
    (a.out/'test_12frames.json').write_text(json.dumps(result,indent=2));print('SHORT_TEST',a.group,a.count,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
