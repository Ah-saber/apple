"""Fit a temporal phase residual from all-frame training GT; validate actual FP16 output."""
import copy
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from target_residual import ROOT,current_setup,PhaseResidual,make_board
from prepare_native_targets import prepare
from streaming_ring_data import InputModel
from causal_detail import GuidanceView,CausalDetail,SensorDetail,reference_board,DetailBoard
from target_validation_v3 import validation,permitted as original_permitted,score,cohorts
from run_reduced_v2 import read_gt
from run_verified import half_board
from gap_metrics_v5 import GapMetrics
from gt_tracks import save_gallery
from paired_timing import paired_timing
from target_training_guidance import balanced_raster
from matched_phase_teacher import reference_field,native_teacher

def permitted(candidate,reference):
    c,r=candidate['frames'],reference['frames']
    return original_permitted(candidate,reference) and c.get('old_position_error_gray',0)<=r.get('old_position_error_gray',0)*1.05+.02

def samples(saved,rng,size=128):
    y,x,h,w=saved['roi']; choices=[]
    pools={'tiny':[],'weak':[],'other':[]}
    for f,regions in saved['regions'].items():
        for region in regions:
            item=region['item']; key='tiny' if item['target_pixels']<=9 else 'weak' if abs(item['gt_gap_gray'])<=15 else 'other'
            cy,cx=region['target'].mean(0)
            for scale in [1,2,4]:
                physical=size*scale
                if min(h,w)<physical: continue
                top=int(np.clip(round((cy-physical/2)/2)*2,(y+1)//2*2,(y+h-physical)//2*2))
                left=int(np.clip(round((cx-physical/2)/2)*2,(x+1)//2*2,(x+w-physical)//2*2))
                pools[key].append((f,top,left,scale))
    for key,cap in [('tiny',144),('weak',48),('other',24)]:
        pool=list(dict.fromkeys(pools[key]))
        if pool: choices.extend(pool[i] for i in rng.choice(len(pool),min(cap,len(pool)),replace=False))
    frames=list(saved['labels'])
    for _ in range(32):
        f=int(rng.choice(frames)); top=int(rng.integers((y+1)//2,(y+h-size)//2+1))*2; left=int(rng.integers((x+1)//2,(x+w-size)//2+1))*2
        choices.append((f,top,left,1))
    return sorted(set(choices))

def run(scene,steps=1200,family='causal_detail_',branch_class=CausalDetail):
    GuidanceView.sensor=branch_class.sensor
    out=ROOT/(family+scene); out.mkdir(exist_ok=False)
    kind,config,df,box_for,count,base,original,profile=current_setup(scene)
    saved=prepare(scene); row=saved['sample']; rng=np.random.default_rng(1009); torch.manual_seed(1009)
    branch=branch_class(count).cuda(); board=reference_board(base,count,kind,box_for(row),scene)
    board.requires_grad_(False)
    data=GuidanceView(df(config,'train'),list(range(9-count,9)))
    choices=defaultdict(list)
    for f,top,left,scale in samples(saved,rng): choices[f].append((top,left,scale))
    values=[]; teacher_differences=[]
    with torch.inference_mode():
        for f,locations in choices.items():
            rr=dict(row,frame_id=f); data.current_record,data.current_crop,data.cache=rr,(0,0,1024,1280),{}
            raw=data.inputs(profile)[None].cuda()
            context,_=data.context_for(rr,**({'crop_tlhw':(0,0,1024,1280)} if kind else {})); context=context[None].cuda()
            if not kind: raw,context=raw.half(),context.half()
            phases=board.phases(raw,context)
            full_reference=reference_field(board,context)
            native=F.pixel_shuffle(phases.float(),2).cpu()
            raw_cpu=raw.float().cpu(); gt=read_gt(config,rr,f)
            for top,left,scale in locations:
                size=128*scale; sl=(slice(top,top+size),slice(left,left+size))
                value=dict(raw=raw_cpu[...,sl[0],sl[1]].clone(),teacher=native[...,sl[0],sl[1]].clone(),gt=torch.from_numpy(gt[sl].copy())[None,None]/255)
                for role in ['target','ghost']:
                    value[role]=torch.from_numpy(balanced_raster(saved['labels'][f][role])[sl].copy())[None,None]
                if scale>1:
                    for name in value: value[name]=F.interpolate(value[name],size=(128,128),mode='area')
                with torch.inference_mode():
                    field=full_reference[...,top//2:(top+size)//2,left//2:(left+size)//2]
                    if scale>1:field=F.interpolate(field.float(),size=(64,64),mode='area')
                    matched=native_teacher(board,value['raw'].cuda(),field).cpu()
                teacher_differences.append(dict(frame=f,scale=scale,mean_inner_difference_gray=float((matched[...,12:-12,12:-12]-value['teacher'][...,12:-12,12:-12]).abs().mean()*255)))
                value['teacher']=matched
                regions=[]
                for region in saved['regions'][f]:
                    t,r=region['target'],region['ring']
                    if not (np.all(t[:,0]>=top+4) and np.all(t[:,0]<top+size-4) and np.all(t[:,1]>=left+4) and np.all(t[:,1]<left+size-4)): continue
                    if not (np.all(r[:,0]>=top) and np.all(r[:,0]<top+size) and np.all(r[:,1]>=left) and np.all(r[:,1]<left+size)): continue
                    def indices(a):
                        a=(a.astype(np.int64)-np.array([top,left]))//scale
                        return torch.from_numpy(np.unique(a[:,0]*128+a[:,1]))
                    ti,ri=indices(t),indices(r)
                    if len(ti)==0 or len(ri)<3: continue
                    g=value['gt'].flatten(); gap=g[ti].mean()-torch.quantile(g[ri],.5)
                    regions.append(dict(target=ti,ring=ri,gap=gap,weight=2 if len(ti)<=9 else 1))
                value['regions']=regions; values.append(value)
            print('TARGET_RESIDUAL_CACHE',scene,f,len(values),flush=True)
    (out/'matched_teacher_audit.json').write_text(json.dumps(teacher_differences,indent=2))
    del board,data,raw,phases,native,raw_cpu,full_reference
    # Clone inference tensors before autograd consumes the cached inputs.
    for v in values:
        for k in ['raw','teacher','gt','target','ghost']: v[k]=v[k].clone().cuda()
        for r in v['regions']:
            for k in ['target','ring','gap']: r[k]=r[k].clone().cuda()
    branch=branch.float().train()
    val=GuidanceView(df(config,'val'),list(range(9-count,9))); vr=next(r for r in val.records if r['scene_id']==scene)
    current=reference_board(base,count,kind,box_for(vr),scene)
    ref=validation({'currentK':InputModel(current,val,profile,kind)},config,val,[scene],box_for)[scene]['currentK']
    logs=[]; best=score(ref); best_step=None
    def check(step):
        nonlocal best,best_step
        candidate=DetailBoard(current,branch).cuda().eval()
        m=validation({'candidate':InputModel(candidate,val,profile,kind)},config,val,[scene],box_for)[scene]['candidate']
        allowed=permitted(m,ref); s=score(m)
        checkpoint=dict(scene=scene,count=count,profile=profile,step=step,branch_class=branch_class.__name__,sensor_inputs=branch_class.sensor,branch={k:v.detach().cpu().clone() for k,v in branch.state_dict().items()},validation=m,reference_validation=ref,eligible=allowed)
        torch.save(checkpoint,out/'last.pt')
        if allowed and s<best-.001:
            best,best_step=s,step; torch.save(checkpoint,out/'best.pt')
        logs.append(dict(step=step,eligible=allowed,score=s,metrics=m))
        (out/'training.json').write_text(json.dumps(dict(scene=scene,steps=steps,samples=len(values),profile=profile,logs=logs,reference_validation=ref,allframe_labels=saved['summary'],trainable='12-channel nonlinear Conv/ReLU/Conv phase detail; temporal DC preserved; reviewed reference frozen',kernel_size=list(branch.conv.kernel_size),exact_ring_gap=True,matched_input='full-frame current streaming preprocessing before crop',GT_scope='training sequence and training spatial ROI',test_used_for_fitting=False,packaged=False,pushed=False),indent=2))
        print('TARGET_RESIDUAL_VALIDATION',scene,step,allowed,s,'reference',score(ref),flush=True)
        del candidate
    optimizer=torch.optim.Adam(branch.parameters(),lr=.002)
    for step in range(1,steps+1):
        chosen=rng.choice(len(values),8)
        batch={k:torch.cat([values[i][k] for i in chosen]) for k in ['raw','teacher','gt','target','ghost']}
        correction=F.pixel_shuffle(branch(batch['raw']),2)
        pred=(batch['teacher']+correction).clamp(0,1); teacher=batch['teacher'].clamp(0,1)
        error=(pred-batch['gt'])*255
        target=batch['target'].abs(); ghost=batch['ghost'].abs()
        def weighted(a,w): return (a*w).sum()/w.sum().clamp_min(1e-8)
        foreground=weighted(error.abs(),target)
        center=error-F.avg_pool2d(error,21,1,10)
        depart=weighted(center.abs(),ghost)
        static=(1-F.max_pool2d(((target+ghost)>0).float(),7,1,3)).clamp(0,1)
        trust=weighted((pred-teacher).abs()*255,static)
        ring_losses=[]; tiny_losses=[]
        for n,i in enumerate(chosen):
            p=pred[n].flatten()
            for region in values[i]['regions']:
                gap=p[region['target']].mean()-torch.quantile(p[region['ring']],.5)
                gap_error=(gap-region['gap']).abs()*255
                ring_losses.append(gap_error)
                if region['weight']==2:tiny_losses.append(gap_error)
        ring=torch.stack(ring_losses).mean() if ring_losses else pred.sum()*0
        tiny=torch.stack(tiny_losses).mean() if tiny_losses else pred.sum()*0
        loss=3*foreground+6*ring+24*tiny+32*depart+12*trust+.5*error.abs().mean()+.05*branch.conv.weight.square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite residual loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(),5); optimizer.step(); branch.project()
        if step in [400,800,steps]: check(step)
    if best_step is None:
        summary=dict(scene=scene,validation_improved=False,reference_includes_reviewed_residual=True,sensor_inputs=branch_class.sensor,reference_validation=ref,validation_trials=logs,automatically_adopted=False,packaged=False,pushed=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'selection.json').write_text(json.dumps(dict(selected=False,reason='No validation improvement passes all quality guards'),indent=2))
        print('CAUSAL_DETAIL_REJECTED',scene,flush=True)
        return
    del values,val,current
    checkpoint=torch.load(out/('best.pt' if best_step is not None else 'last.pt'),map_location='cpu',weights_only=False)
    branch.load_state_dict(checkpoint['branch']); branch.eval()
    data=GuidanceView(df(config,'test'),list(range(9-count,9))); row=next(r for r in data.records if r['scene_id']==scene)
    current=reference_board(base,count,kind,box_for(row),scene)
    candidate=DetailBoard(current,branch).cuda().eval(); control=original if kind else half_board(original,box_for(row).cuda())
    boards=dict(control9=control,currentK=current,candidate=candidate)
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization; key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']; frames=int(norm.by_key[key]['frames'])
    metric=GapMetrics(row['eval_crop_tlhw'],boards); metric.prepare_background([read_gt(config,row,f) for f in range(frames)])
    with torch.inference_mode():
        for f in range(frames):
            rr=dict(row,frame_id=f); data.current_record,data.current_crop,data.cache=rr,(0,0,1024,1280),{}
            raw=data.inputs(profile)[None].cuda(); context,_=data.context_for(rr,**({'crop_tlhw':(0,0,1024,1280)} if kind else {})); context=context[None].cuda()
            old=data.source.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
            if not kind: raw,context,old=raw.half(),context.half(),old.half()
            predictions={name:model(old if name=='control9' else raw,context)[0,0,::3,::3].float().cpu().numpy() for name,model in boards.items()}
            metric.update(f,read_gt(config,row,f),predictions)
    timing=paired_timing(dict(currentK=current,candidate=candidate),[(raw,context)])
    for stages in timing.values():
        for v in stages.values(): v.pop('samples_ms',None)
    report=metric.report(); report.update(sensor_inputs=branch_class.sensor,additional_sensor_channels=3 if branch_class.sensor else 0,reference_includes_reviewed_residual=True,scene=scene,profile=profile,channels=count,checkpoint_step=checkpoint['step'],validation_eligible=checkpoint['eligible'],validation_improved=best_step is not None,timing=timing,cohorts=cohorts(report['targets'],boards),NPU_measured=False,automatically_adopted=False,packaged=False,pushed=False,test_used_for_selection=False)
    (out/'full_field_results.json').write_text(json.dumps(report,indent=2)); save_gallery(metric,out)
    summary={k:v for k,v in report.items() if k not in ['frames','targets']}; (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'selection.json').write_text(json.dumps(dict(selected=best_step is not None,validation_eligible=checkpoint['eligible'],diagnostic_checkpoint=checkpoint['step'],automatically_adopted=False),indent=2))
    print('TARGET_RESIDUAL_COMPLETE',scene,json.dumps(summary),flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--scene',required=True)
    p.add_argument('--sensor',action='store_true')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--out')
    a=p.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.scene,a.steps,'balanced_detail_',SensorDetail if a.sensor else CausalDetail)

if __name__=='__main__': main()



