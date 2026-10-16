"""Train GT-confirmed foreground and departed-position corrections per scene."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_reduced_v2 import ROOT,R2,load_group,ShortHistory,night_source,make_board,fold_conv,read_gt
from run_verified import Adapter,validation,permitted,half_board
from run_night_precision import noise_filter
from gap_metrics_v5 import GapMetrics
from guided_motion import GuidedMotion,GuidedBoard
from direct_motion import DirectBoard
from run_speed import measure
from gt_tracks import save_gallery
RUN_ROOT=Path(__file__).resolve().parents[1]

def score(m):
    f,t=m['frames'],m['targets']
    return .25*f['mae_gray']+.25*f.get('weak_structure_error_gray',0)+2*f.get('old_position_signed_ghost_gray',0)+(t.get('gap_error_gray',0) if t else 0)

def setup(scene):
    kind=scene.removeprefix('night_') if scene.startswith('night_') else None
    group='day' if scene=='day_normal' else 'light' if scene in ['weather_light','weather_medium'] else 'heavy'
    if kind:
        source=night_source(kind)
        from run_motion import V13
        config=json.loads((V13/'night_calibration_final'/scene/'manifest.json').read_text())['config']
        from ir_sr.training import dataset_for_config
        df=dataset_for_config;box_for=lambda r:torch.zeros(1,4)
        v=json.loads((RUN_ROOT/('precision_'+kind)/'validation.json').read_text())
        candidates=[n for n in ['fp16_5','noise5','fp16_7','noise7'] if v['eligible'].get(n)]
        choice=min(candidates,key=lambda n:(5 if n.endswith('5') else 7,score(v['metrics'][scene][n]))) if candidates else 'fp16_7'
        count=int(choice[-1]);indices=list(range(9-count,9))
        base=make_board(source,count,torch.zeros(1,4).cuda(),kind)
        if choice.startswith('noise'):
            front,_=noise_filter(source.front.temporal_conv,base.source.front.temporal_conv)
            base.source.front.temporal_conv=front
        base.source.front.half();base.requires_grad_(False)
        original=make_board(source,9,torch.zeros(1,4).cuda(),kind).requires_grad_(False)
        return kind,group,config,df,box_for,choice,count,indices,base,original,None
    _,source,config,_,_,df,box_for=load_group(group)
    source=source.cuda().float().eval().requires_grad_(False)
    v=json.loads((RUN_ROOT/('verified_'+scene)/'validation.json').read_text())
    candidates=[n for n in ['trained3','recent5','grid5'] if v['eligible'].get(n)]
    choice=min(candidates,key=lambda n:(3 if n=='trained3' else 5,score(v['metrics'][scene][n]))) if candidates else 'grid5'
    if choice=='trained3':
        short=ShortHistory(source,3,'half').cuda().float()
        short.load_state_dict(torch.load(R2/f'half3_{group}_3/best.pt',map_location='cpu',weights_only=False)['model'])
        candidate=short.source.eval().requires_grad_(False);count=3;indices=[6,7,8]
    else:
        candidate=copy.deepcopy(source)
        candidate.quarter.front.first=fold_conv(candidate.quarter.front.first,5)
        candidate.current_only=False;count=5;indices=[1,3,5,7,8] if choice=='grid5' else [4,5,6,7,8]
    return None,group,config,df,box_for,choice,count,indices,candidate,source,source

def raster(points):
    value=np.zeros((1024,1280),np.int8)
    if len(points):value[points[:,0],points[:,1]]=points[:,2]
    return value

def crops(saved,size,rng):
    roi=saved['roi'];ry,rx,rh,rw=roi
    candidates={'target':[],'ghost':[]}
    labels=saved['labels']
    for frame,m in labels.items():
        if frame%4 or frame==0:continue
        for role in candidates:
            a=raster(m[role])
            _,_,stats,centers=cv2.connectedComponentsWithStats((a!=0).astype(np.uint8),8)
            for center in centers[1:]:
                x,y=center
                top=int(np.clip(round((y-size/2)/2)*2,(ry+1)//2*2,(ry+rh-size)//2*2))
                left=int(np.clip(round((x-size/2)/2)*2,(rx+1)//2*2,(rx+rw-size)//2*2))
                candidates[role].append((frame,top,left))
    chosen=[]
    for role in candidates:
        pool=list(dict.fromkeys(candidates[role]))
        if pool:
            ids=rng.choice(len(pool),min(24,len(pool)),replace=False)
            chosen.extend(pool[i] for i in ids)
    frames=[f for f in labels if f%4==0]
    for _ in range(16):
        f=int(rng.choice(frames))
        top=int(rng.integers((ry+1)//2,(ry+rh-size)//2+1))*2
        left=int(rng.integers((rx+1)//2,(rx+rw-size)//2+1))*2
        chosen.append((f,top,left))
    return list(dict.fromkeys(chosen))

def objective(pred,gt,teacher,target,ghost):
    border=12
    mask=target.abs();depart=ghost.abs()
    e=(pred-gt)*255
    gap=(pred-F.avg_pool2d(pred,7,1,3))-(gt-F.avg_pool2d(gt,7,1,3))
    centered=e-F.avg_pool2d(e,21,1,10)
    region=(slice(None),slice(None),slice(border,-border),slice(border,-border))
    def weighted(value,weight):
        value,weight=value[region],weight[region]
        return (value*weight).sum()/weight.sum().clamp_min(1)
    pixels=weighted(e.abs(),mask)
    contrast=weighted(gap.abs()*255,mask)
    old=weighted(F.relu(centered*ghost.sign()),depart)
    old_abs=weighted(centered.abs(),depart)
    static=(1-F.max_pool2d(torch.maximum(mask,depart),7,1,3)).clamp(0,1)
    imitate=weighted((pred-teacher).abs()*255,static)
    teach_gap=(pred-F.avg_pool2d(pred,7,1,3))-(teacher-F.avg_pool2d(teacher,7,1,3))
    weak=weighted(teach_gap.abs()*255,static)
    pair=e.reshape(-1,2,1,e.shape[-2],e.shape[-1])
    pairmask=torch.maximum(mask[::2],mask[1::2])
    temporal=weighted((pair[:,1]-pair[:,0]).abs(),pairmask)
    loss=4*pixels+6*contrast+8*old+2*old_abs+3*imitate+weak+2*temporal
    return loss,{'foreground_gray':float(pixels.detach()),'gap_gray':float(contrast.detach()),
                 'ghost_gray':float(old.detach()),'static_teacher_gray':float(imitate.detach())}

def run(scene,out,steps):
    kind,group,config,df,box_for,choice,count,indices,base,original,_=setup(scene)
    saved=torch.load(RUN_ROOT/('guided_labels_'+scene)/'labels.pt',map_location='cpu',weights_only=False)
    rng=np.random.default_rng(1008);torch.manual_seed(1008)
    size=256 if kind else 128
    selections=crops(saved,size,rng)
    data=df(config,'train');row=saved['sample']
    assert row['split']=='train'
    values=[];native_cache={}
    for frame,top,left in selections:
        tensors={k:[] for k in ['raw','gt','baseline','teacher','target','ghost']}
        for f in [max(0,frame-1),frame]:
            rr=dict(row,frame_id=f)
            crop=(top,left,size,size)
            raw9=data.normalized_stack(rr,crop)[None].cuda().float()
            context,box=data.context_for(rr,crop_tlhw=crop)
            context,box=context[None].cuda().float(),box[None].cuda().float()
            with torch.inference_mode():
                if kind:
                    if f not in native_cache:
                        full=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
                        c,_=data.context_for(rr,crop_tlhw=(0,0,1024,1280));c=c[None].cuda()
                        pair=[]
                        for b,ids in [(base,indices),(original,list(range(9)))]:
                            phases=DirectBoard.night_phases(SimpleNamespace(board=b,kind=kind),full[:,ids].contiguous(),c)
                            pair.append(F.pixel_shuffle(phases.float(),2).cpu())
                        native_cache[f]=pair
                    baseline,teacher=[v[:,:,top:top+size,left:left+size] for v in native_cache[f]]
                else:
                    baseline=base.native(raw9[:,indices],context,box).cpu()
                    teacher=original.native(raw9,context,box).cpu()
            gt=torch.from_numpy(read_gt(config,rr,f)[top:top+size,left:left+size])[None,None]/255
            label=saved['labels'].get(f,{'target':np.empty((0,3),np.int16),'ghost':np.empty((0,3),np.int16)})
            tensors['raw'].append(raw9[:,indices].cpu())
            tensors['gt'].append(gt);tensors['baseline'].append(baseline);tensors['teacher'].append(teacher.clamp(0,1))
            for role in ['target','ghost']:
                a=raster(label[role])[top:top+size,left:left+size].astype(np.float32)
                tensors[role].append(torch.from_numpy(a)[None,None])
        values.append({k:torch.cat(v) for k,v in tensors.items()})
    del native_cache
    branch=GuidedMotion(count).cuda().float()
    branch.calibrate([torch.cat([v['raw'] for v in values[:8]]).cuda()])
    val=df(config,'val')
    vrow=next(r for r in val.records if r['scene_id']==scene)
    def board_for(row,candidate=True):
        board=copy.deepcopy(base) if kind else half_board(base,box_for(row).cuda())
        return GuidedBoard(board,branch,count,kind).cuda().eval() if candidate else board
    control=original if kind else half_board(original,box_for(vrow).cuda())
    reference=Adapter(control,list(range(9)))
    baseline_metrics=validation({'control9':reference},config,val,[scene],box_for)
    best=float('inf');logs=[]
    def check(step):
        nonlocal best
        board=board_for(vrow)
        result=validation({'guided':Adapter(board,indices)},config,val,[scene],box_for)
        m=result[scene]['guided'];ref=baseline_metrics[scene]['control9']
        ok=permitted(m,ref);value=score(m)
        ck={'branch':{k:v.detach().cpu() for k,v in branch.state_dict().items()},'scene':scene,
            'count':count,'indices':indices,'source_choice':choice,'step':step,'eligible':ok,
            'score':value,'validation':result,'reference_validation':baseline_metrics}
        torch.save(ck,out/'last.pt')
        if ok and value<best:best=value;torch.save(ck,out/'best.pt')
        record={'step':step,'eligible':ok,'score':value,'metrics':result};logs.append(record)
        print('GUIDED_VALIDATION',scene,step,ok,value,flush=True)
        del board
    check(0)
    opt=torch.optim.Adam([p for p in branch.parameters() if p.requires_grad],lr=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-6)
    for step in range(1,steps+1):
        chosen=rng.choice(len(values),4)
        batch={k:torch.cat([values[i][k] for i in chosen]).cuda() for k in values[0]}
        pred=(batch['baseline']+branch(batch['raw'],batch['baseline'])).clamp(0,1)
        loss,metrics=objective(pred,batch['gt'],batch['teacher'],batch['target'],batch['ghost'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite guided loss')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(branch.parameters(),5)
        opt.step();branch.project();schedule.step()
        if step%500==0 or step==steps:
            check(step)
            (out/'training.json').write_text(json.dumps({'scene':scene,'count':count,'source_choice':choice,
                'steps':steps,'samples':len(values),'logs':logs,'loss':metrics,
                'training_scope':'training GT and spatial ROI only; GT-confirmed foreground and historical positions',
                'static_teacher':'original nine-frame source','trainable':'six motion, local-contrast and gate scalars; source frozen',
                'checkpoint_selection':'validation only; actual FP16 phase graph and GT departure/arrival masks',
                'packaged':False,'pushed':False,'hash_checks':False},indent=2))
    if not (out/'best.pt').exists():
        (out/'selection.json').write_text(json.dumps({'selected':False,'reason':'No validation candidate passes quality guards'}));return
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    (out/'selection.json').write_text(json.dumps({'selected':ck['step']>0,'step':ck['step'],'source_choice':choice,'count':count}))
    if ck['step']==0:return
    branch.load_state_dict(ck['branch'])
    data=df(config,'test');row=next(r for r in data.records if r['scene_id']==scene)
    controls={'control9':original if kind else half_board(original,box_for(row).cuda()),'guided':board_for(row)}
    counts={'control9':9,'guided':count}
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization;key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    nframes=int(norm.by_key[key]['frames'])
    metric=GapMetrics(row['eval_crop_tlhw'],controls)
    metric.prepare_background([read_gt(config,row,f) for f in range(nframes)])
    with torch.inference_mode():
        for f in range(nframes):
            rr=dict(row,frame_id=f);raw=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(rr,**({'crop_tlhw':(0,0,1024,1280)} if kind else {}));context=context[None].cuda()
            if not kind:raw,context=raw.half(),context.half()
            ps={n:b(raw if n=='control9' else raw[:,indices].contiguous(),context)[0,0,::3,::3].float().cpu().numpy() for n,b in controls.items()}
            metric.update(f,read_gt(config,row,f),ps)
    from run_motion import V13
    directory=V13/('night_calibration_final' if kind else 'deployment_corrected')/scene
    prepared=[]
    for p in sorted(directory.glob('test_*.npz')):
        with np.load(p,allow_pickle=False) as a:args=tuple(torch.from_numpy(a[k]).cuda() for k in ['nine_raw','reference_thumb'])
        if not kind:args=tuple(x.half() for x in args)
        elif kind=='ordinary':args=(args[0].half(),args[1])
        prepared.append(args)
    timing={n:measure(b,[(x if n=='control9' else x[:,indices].contiguous(),c) for x,c in prepared],50,300) for n,b in controls.items()}
    report=metric.report();report.update(scene=scene,source_choice=choice,validation_selected_step=ck['step'],
        actual_input_channels=counts,timing=timing,automatically_adopted=False,NPU_measured=False)
    (out/'full_field_results.json').write_text(json.dumps(report,indent=2));save_gallery(metric,out)
    summary={k:report[k] for k in ['scene','source_choice','validation_selected_step','summary','target_observations','GT_tracks','actual_input_channels']}
    summary['timing']={n:{k:{j:v for j,v in m.items() if j in ['mean_ms','p95_ms']} for k,m in t.items()} for n,t in timing.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('GUIDED_COMPLETE',scene,ck['step'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.scene,a.out,a.steps)
if __name__=='__main__':main()

