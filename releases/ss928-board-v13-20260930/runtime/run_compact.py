"""Train on existing train caches, select on val, then compare complete test videos."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

ROOT = Path('/data/zhangbenzhuang/huawei_sr')
CODE = ROOT/'code/worktrees/ss928-five-scene-nine-20260929'
OLD = ROOT/'runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS'
GROUPS = {
    'day': {'scenes':['day_normal'], 'teacher':'DAY-FULLNIGHT-LR1E4',
            'baseline':'HALF-DAY-NINE-TEMPORAL', 'cache':'QUARTER-DAY', 'width':24, 'depth':2, 'raw_skip':True},
    'light': {'scenes':['weather_light','weather_medium'], 'teacher':'LIGHT-FULLNIGHT-20K',
              'baseline':'HALF-LIGHT-TEMPORAL', 'cache':'QUARTER-LIGHT', 'width':16, 'depth':1, 'raw_skip':False},
    'heavy': {'scenes':['weather_heavy','weather_heavy_c32'], 'teacher':'HEAVY-FULLNIGHT-20K',
              'baseline':'HALF-HEAVY-TEMPORAL', 'cache':'QUARTER-HEAVY', 'width':16, 'depth':1, 'raw_skip':False},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def setup(group):
    sys.path[:0] = [str(CODE/'src'), str(OLD)]
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from half_student import make_student
    from train_quarter_student import BoxQuarter, box_for
    from compact_model import CompactModel
    spec = GROUPS[group]
    teacher = ROOT/'runs'/('SS928-FIVE-SCENE-NINE-SCRATCH-20260929-'+spec['teacher'])/'checkpoints/best.pt'
    state = torch.load(teacher, map_location='cpu', weights_only=False)
    reference = inference_model(state['config'], state['model']).global_reference
    model = CompactModel(reference, spec['width'], spec['depth'], spec['raw_skip'],spec.get('late_reference',True))
    base_path = ROOT/'runs'/('SS928-FIVE-SCENE-'+spec['baseline']+'-20260929')
    base_path /= 'best.pt' if group == 'day' else 'average05.pt'
    ck = torch.load(base_path, map_location='cpu', weights_only=False)
    baseline = BoxQuarter(make_student(copy.deepcopy(reference),ck),current_only=ck.get('current_only',False))
    baseline.load_state_dict(ck['model'], strict=True)
    return model, baseline, state['config'], teacher, base_path, dataset_for_config, box_for


def score(model, dataset, config, scenes, box_for, limit=None):
    values = {s:[] for s in scenes}
    model.eval()
    with torch.inference_mode():
        for row in dataset.records:
            scene = row['scene_id']
            if scene not in values or (limit and len(values[scene]) >= limit):
                continue
            x = dataset.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            c,_ = dataset.context_for(row)
            pred = model.native(x,c[None].cuda(),box_for(row).cuda()).float().clamp(0,1)
            gt = np.array(Image.open(Path(config['data_root'])/row['target']['path']),dtype=np.float32)/255
            y,x0,h,w = row['eval_crop_tlhw']; y+=3; x0+=3; h-=6; w-=6
            err = pred[0,0,y:y+h,x0:x0+w]-torch.from_numpy(gt[y:y+h,x0:x0+w]).cuda()
            values[scene].append({'sample_id':row['sample_id'],
                'psnr_db':-10*np.log10(max(float(err.square().mean()),1e-12)),
                'mae_gray':float(err.abs().mean())*255,'bias_gray':float(err.mean())*255})
    assert all(values.values())
    return {'summary':{s:{'psnr_db':float(np.mean([r['psnr_db'] for r in rows])),
                         'mae_gray':float(np.mean([r['mae_gray'] for r in rows]))}
                       for s,rows in values.items()}, 'samples':values}


def train(group, out, steps):
    model,base,config,teacher,base_path,df,box_for = setup(group)
    spec = GROUPS[group]
    del base
    cache = ROOT/'runs'/('SS928-FIVE-SCENE-'+spec['cache']+'-20260929')/'train_cache.pt'
    saved = torch.load(cache,map_location='cpu',weights_only=False)
    data, metadata = saved['values'],saved['metadata']
    assert all(r['scene'] in spec['scenes'] for r in metadata)
    model = model.cuda().to(memory_format=torch.channels_last)
    if spec.get('resume'):
        resumed=torch.load(spec['resume'],map_location='cpu',weights_only=False)
        model.load_state_dict(resumed['model'],strict=True)
    model.reference.encoder.requires_grad_(False)
    if hasattr(model.reference,'pyramid'):
        model.reference.pyramid.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=5e-5 if spec.get('temporal') else 2e-4,weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-6)
    val = df(config,'val')
    rng = np.random.default_rng(930)
    logs=[]; best=-np.inf; started=time.perf_counter()
    for step in range(1,steps+1):
        ids=rng.integers(0,len(metadata),4)
        stack=data['stack'][ids].flatten(0,1).cuda().float()
        context=data['context'][ids].flatten(0,1).cuda().float()
        box=data['box'][ids].flatten(0,1).cuda().float()
        gt=data['gt'][ids].flatten(0,1).cuda().float()
        target=data['teacher'][ids].flatten(0,1).cuda().float()
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.native(stack,context,box).float().clamp(0,1)
            error=(pred-gt)*255
            e=error[:,:,12:-12,12:-12]
            pixel=e.abs().mean()
            imitate=((pred-target)*255)[:,:,12:-12,12:-12].abs().mean()
            pair=e.reshape(4,2,1,104,104)
            temporal=(pair[:,1]-pair[:,0]).abs()
            delta=(gt[1::2]-gt[::2]).abs()[:,:,12:-12,12:-12]*255
            weak=(delta>=1)&(delta<3); moving=delta>=3
            weak_loss=(temporal*weak).sum()/weak.sum().clamp_min(1)
            move_loss=(temporal*moving).sum()/moving.sum().clamp_min(1)
            # Penalize spatial phase bias in train data; this does not model SDK quantization.
            phase_means=F.pixel_unshuffle(e,4).mean((-2,-1))
            phase_loss=(phase_means-phase_means.mean(1,keepdim=True)).abs().mean()
            loss=(pixel+.35*imitate+2.*temporal.mean()+1.5*weak_loss+2.5*move_loss+.1*phase_loss
                  if spec.get('temporal') else
                  pixel+.35*imitate+.75*temporal.mean()+.75*weak_loss+1.5*move_loss+.1*phase_loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5); opt.step(); schedule.step()
        if step==1 or step%1000==0:
            log={'step':step,'loss':float(loss),'train_mae_gray':float(pixel),
                 'elapsed_seconds':time.perf_counter()-started}
            if step%2000==0 or step==steps:
                scores=score(model,val,config,spec['scenes'],box_for)
                log['validation']=scores['summary']
                mean=float(np.mean([v['psnr_db'] for v in scores['summary'].values()]))
                if mean>best:
                    best=mean
                    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'spec':spec,'step':step,'val':scores,'config_checkpoint':str(teacher),
                        'seed':930,'train_cache':str(cache),'test_used_for_selection':False},out/'best.pt')
            logs.append(log)
            (out/'training.json').write_text(json.dumps({'spec':spec,'seed':930,'steps':steps,
                'cache':str(cache),'cache_sha256':sha(cache),'metadata':metadata,'logs':logs,
                'source_commit':'0c1f674d7b5f1d1aed8e75ac64082c97e0dedc9e'},indent=2))
            print(group,json.dumps(log),flush=True)
    return out/'best.pt'


def temporal_metrics(pred, previous, gt, prev_gt):
    change=gt-prev_gt
    gradient=np.maximum.reduce([np.abs(gt-np.roll(gt,i,axis=a)) for a in (0,1) for i in (-1,1)])
    static=(np.abs(change)<=1)&(gradient<=5)
    moving=np.abs(change)>=3
    weak=(np.abs(change)>=1)&(np.abs(change)<3)&(gradient>=5)
    delta=pred-previous
    r={}
    if static.any(): r['static_residual_change_gray']=float(np.abs(delta-change)[static].mean())
    if moving.any(): r['motion_response_ratio']=float((np.sign(change[moving])*delta[moving]).mean()/np.abs(change[moving]).mean())
    if weak.any(): r['weak_structure_error_gray']=float(np.abs(delta-change)[weak].mean())
    return r


def evaluate(group,out):
    model,baseline,config,teacher,base_path,df,box_for=setup(group)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    model=model.cuda().eval().to(memory_format=torch.channels_last)
    baseline=baseline.cuda().eval().to(memory_format=torch.channels_last)
    data=df(config,'test'); scenes=GROUPS[group]['scenes']
    report={'candidate':str(out/'best.pt'),'candidate_sha256':sha(out/'best.pt'),
            'baseline':str(base_path),'baseline_sha256':sha(base_path),
            'source_test':score(model,data,config,scenes,box_for),
            'baseline_test':score(baseline,data,config,scenes,box_for),'sequences':{}}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
    for scene in scenes:
        row=next(r for r in sorted(data.records,key=lambda r:r['frame_id']) if r['scene_id']==scene)
        video=out/f'{scene}_full120.mp4'
        writer=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','5120x1080',
            '-r','12','-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast',
            '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(video)],stdin=subprocess.PIPE)
        metrics=[]; previous=None
        try:
            with torch.inference_mode():
                for frame in range(120):
                    rr=dict(row,frame_id=frame)
                    x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
                    c,_=data.context_for(rr); b=box_for(rr).cuda(); c=c[None].cuda()
                    pred=model.native(x,c,b).float().clamp(0,1)[0,0].cpu().numpy()*255
                    old=baseline.native(x,c,b).float().clamp(0,1)[0,0].cpu().numpy()*255
                    path=(Path(config['data_root'])/rr['target']['path']).with_name(f'{frame:06d}.png')
                    gt=np.array(Image.open(path),dtype=np.float32)
                    canvas=Image.new('RGB',(5120,1080),'#161b22'); draw=ImageDraw.Draw(canvas)
                    for i,(a,title) in enumerate(zip((x[0,-1].cpu().numpy()*255,gt,old,pred),
                                ('已校正原始输入','GT','v0.12半网格','本轮四分之一网格'))):
                        canvas.paste(Image.fromarray(np.rint(a.clip(0,255)).astype(np.uint8)).convert('RGB'),(i*1280,56))
                        draw.text((i*1280+12,10),f'{scene} | {title} | {frame:03d}',font=font,fill='white')
                    writer.stdin.write(np.asarray(canvas).tobytes())
                    y,x0,h,w=rr['eval_crop_tlhw']; y+=3; x0+=3; h-=6; w-=6
                    p=pred[y:y+h,x0:x0+w]; o=old[y:y+h,x0:x0+w]; g=gt[y:y+h,x0:x0+w]
                    item={'frame':frame,'new_mae_gray':float(np.abs(p-g).mean()),
                          'old_mae_gray':float(np.abs(o-g).mean()),'new_bias_gray':float((p-g).mean()),
                          'old_bias_gray':float((o-g).mean()),'new_contrast_error_gray':float(p.std()-g.std()),
                          'old_contrast_error_gray':float(o.std()-g.std())}
                    if previous:
                        for prefix,a,pa in [('new',p,previous[0]),('old',o,previous[1])]:
                            item.update({prefix+'_'+k:v for k,v in temporal_metrics(a,pa,g,previous[2]).items()})
                    metrics.append(item); previous=(p.copy(),o.copy(),g.copy())
                    if frame%20==0: print('VIDEO',scene,frame,flush=True)
        finally:
            writer.stdin.close()
            assert writer.wait()==0
        summary={k:float(np.mean([r[k] for r in metrics if k in r])) for k in metrics[-1] if k!='frame'}
        for prefix in ['new','old']:
            for field in ['bias_gray','contrast_error_gray']:
                summary[prefix+'_'+field+'_std']=float(np.std([r[prefix+'_'+field] for r in metrics]))
        report['sequences'][scene]={'sequence_id':row['sequence_id'],'frames':120,
            'eval_crop_tlhw':row['eval_crop_tlhw'],'video':str(video),'video_sha256':sha(video),
            'summary':summary,'per_frame':metrics}
        print('QUALITY',scene,json.dumps(summary),flush=True)
        (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--group',choices=GROUPS,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--evaluate-only',action='store_true'); p.add_argument('--fused-reference',action='store_true')
    p.add_argument('--resume',type=Path); p.add_argument('--temporal',action='store_true'); a=p.parse_args()
    GROUPS[a.group]['late_reference']=not a.fused_reference
    GROUPS[a.group]['resume']=str(a.resume) if a.resume else None
    GROUPS[a.group]['temporal']=a.temporal
    a.out.mkdir(parents=True,exist_ok=False) if not a.evaluate_only else None
    torch.set_num_threads(2); torch.manual_seed(930)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    if not a.evaluate_only: train(a.group,a.out,a.steps)
    evaluate(a.group,a.out)


if __name__=='__main__': main()
