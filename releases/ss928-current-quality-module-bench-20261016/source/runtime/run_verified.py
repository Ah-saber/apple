"""Recheck full sequences using GT departure/arrival tracks. No training or export."""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from run_reduced_v2 import (ROOT,R2,V13,load_group,ShortHistory,night_source,make_board,read_gt,
                            BoardHalfOptimized,structural)
from gap_metrics_v5 import GapMetrics
from direct_motion import DirectMotion,DirectBoard
from linear_motion import LinearMotion,LinearBoard
from gt_tracks import save_gallery
from train_innovation_half import validation

validation.__globals__['GapMetrics']=GapMetrics

class Adapter(nn.Module):
    def __init__(self,board,indices):
        super().__init__()
        self.board=board;self.indices=indices
    def native(self,raw,context,box):
        return self.board(raw[:,self.indices].contiguous(),context)[...,::3,::3].float()/255

def permitted(candidate,reference):
    c,r=candidate['frames'],reference['frames']
    ok=c['psnr_db']>=r['psnr_db']-.05
    for key,scale,margin in [('static_temporal_error_gray',1.1,.02),
                              ('weak_structure_error_gray',1.05,.02),
                              ('old_position_signed_ghost_gray',1,.02)]:
        if key in c and key in r:ok=ok and c[key]<=r[key]*scale+margin
    if candidate['targets'] and reference['targets']:
        ok=ok and candidate['targets']['gap_error_gray']<=reference['targets']['gap_error_gray']+.02
    return bool(ok)

def half_board(source,box):
    board=BoardHalfOptimized(source,box).cuda().half().eval()
    structural.replace_samples(board,8)
    board.rows=structural.ChannelCopyOutput().cuda().half()
    return board

def run(scene,out):
    kind=scene.removeprefix('night_') if scene.startswith('night_') else None
    group='day' if scene=='day_normal' else 'light' if scene in ['weather_light','weather_medium'] else 'heavy'
    if kind:
        source=night_source(kind)
        config=json.loads((V13/'night_calibration_final'/scene/'manifest.json').read_text())['config']
        from ir_sr.training import dataset_for_config
        df=dataset_for_config;box_for=lambda r:torch.zeros(1,4)
        short=None
    else:
        _,source,config,_,_,df,box_for=load_group(group)
        short=ShortHistory(source,3,'half')
        short.load_state_dict(torch.load(R2/f'half3_{group}_3/best.pt',map_location='cpu',weights_only=False)['model'])
    indices={'original9':list(range(9)), 'recent5':list(range(4,9)),
             'grid5':[1,3,5,7,8]}
    if not kind:indices['trained3']=[6,7,8]
    branches={}
    for label,cls,prefix,count in [('direct',DirectMotion,'direct',5),('linear',LinearMotion,'linear',3 if not kind else 5)]:
        p=Path(__file__).resolve().parents[1]/(prefix+'_'+scene+str(count))/'best.pt'
        if not p.exists():continue
        ck=torch.load(p,map_location='cpu',weights_only=False)
        if ck['step']==0:continue
        weights=ck.get('branch')
        if weights is None:weights={k.removeprefix('branch.'):v for k,v in ck['model'].items() if k.startswith('branch.')}
        branch=cls(count).cuda().float().eval();branch.load_state_dict(weights)
        branches[label]=(branch,p,ck['step'])
        indices[label]=list(range(9-count,9))
    def boards_at(row):
        box=box_for(row).cuda()
        original=make_board(source,9,box,kind)
        reduced=make_board(source,5,box,kind)
        boards={'original9':original,'recent5':reduced,'grid5':reduced}
        if not kind:boards['trained3']=half_board(short.source,box)
        for name,(branch,p,step) in branches.items():
            if kind:base=reduced;source_count=5
            else:base=boards['trained3'];source_count=3
            cls=LinearBoard if name=='linear' else DirectBoard
            boards[name]=cls(base,branch,source_count,kind).cuda().eval()
        return boards
    val=df(config,'val')
    vrow=next(r for r in val.records if r['scene_id']==scene)
    vboards=boards_at(vrow)
    models={n:Adapter(b,indices[n]) for n,b in vboards.items()}
    result=validation(models,config,val,[scene],box_for)
    eligible={n:permitted(m,result[scene]['original9']) for n,m in result[scene].items()}
    (out/'validation.json').write_text(json.dumps({'scene':scene,'metrics':result,'eligible':eligible,
        'selection':'validation only; actual FP16 display graph; GT departure and arrival masks'},indent=2))
    print('VERIFIED_VALIDATION',scene,json.dumps(eligible),flush=True)
    del models,vboards
    data=df(config,'test');row=next(r for r in data.records if r['scene_id']==scene)
    boards=boards_at(row)
    # All controls and reduced-input variants are measured under identical GT masks.
    # Results are diagnostic; rejected validation models cannot be selected using test.
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization
    key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    nframes=int(norm.by_key[key]['frames'])
    metric=GapMetrics(row['eval_crop_tlhw'],boards)
    metric.prepare_background([read_gt(config,row,f) for f in range(nframes)])
    with torch.inference_mode():
        for f in range(nframes):
            rr=dict(row,frame_id=f)
            raw=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(rr,**({'crop_tlhw':(0,0,1024,1280)} if kind else {}))
            context=context[None].cuda()
            if not kind:raw,context=raw.half(),context.half()
            ps={n:b(raw[:,indices[n]].contiguous(),context)[0,0,::3,::3].float().cpu().numpy() for n,b in boards.items()}
            metric.update(f,read_gt(config,row,f),ps)
            if f%30==0:print('VERIFIED_FRAME',scene,f,flush=True)
    report=metric.report()
    report.update(scene=scene,validation_eligible=eligible,actual_input_channels={n:len(i) for n,i in indices.items()},
        input_indices_from_causal_nine=indices,grid5_offsets=[-7,-5,-3,-1,0],automatically_adopted=False,
        test_used_for_selection=False,NPU_measured=False,packaged=False,pushed=False)
    (out/'full_field_results.json').write_text(json.dumps(report,indent=2))
    summary={k:report[k] for k in ['scene','target_observations','GT_tracks','summary','validation_eligible','actual_input_channels','input_indices_from_causal_nine']}
    # Existing GPU timings remain applicable. Graphs and weights are unchanged; GT masks changed.
    timings={}
    old=Path(__file__).resolve().parents[1]/('reduced_'+('night' if kind else group))/scene/'summary.json'
    if old.exists():
        previous=json.loads(old.read_text())
        for name,original_name in [('original9','original9'),('recent5','fold5'),('grid5','fold5'),('trained3','trained3')]:
            if name in boards and original_name in previous.get('timing',{}):
                timings[name]={'source':str(old),'timing':previous['timing'][original_name],
                               'reused_reason':'same deployed graph and weights; input index choice outside timed model'}
    for name,(branch,p,step) in branches.items():
        d=p.parent if kind else p.parent/scene
        s=json.loads((d/'summary.json').read_text())
        timings[name]={'source':str(d/'summary.json'),'timing':s['timing']['gap_candidate']}
    summary['existing_GPU_timing']=timings
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    save_gallery(metric,out)
    print('VERIFIED_COMPLETE',scene,report['target_observations'],json.dumps(eligible),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.scene,a.out)

if __name__=='__main__':main()

