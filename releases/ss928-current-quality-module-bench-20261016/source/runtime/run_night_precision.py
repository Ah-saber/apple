"""Night FP16 front and five/seven-frame noise compensation; no training or export."""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from run_reduced_v2 import ROOT,V13,night_source,make_board,read_gt
from run_speed import measure
from run_verified import Adapter,permitted,validation
from gap_metrics_v5 import GapMetrics
from gt_tracks import save_gallery

def noise_filter(old,short):
    w9=old.weight.detach().float()[:,:,0,0]
    w=short.weight.detach().float()[:,:,0,0]
    e9=(w9*w9).sum(1)
    current=w[:,-1].square()
    past=w[:,:-1].square().sum(1).clamp_min(1e-12)
    variance=((e9-current)/past).clamp(1/9,1)
    center=(1+torch.sqrt((72*variance-8).clamp_min(0)))/9
    kernel=((1-center)/8)[:,None,None,None].expand(-1,short.in_channels,3,3).clone()
    kernel[:,:,1,1]=center[:,None]
    kernel[:,-1].zero_();kernel[:,-1,1,1]=1
    result=nn.Conv2d(short.in_channels,short.out_channels,3,padding=1,
                     bias=short.bias is not None,device=short.weight.device,dtype=torch.float32)
    with torch.no_grad():
        result.weight.copy_(kernel*w[:,:,None,None])
        if short.bias is not None:result.bias.copy_(short.bias)
    return result,{'center':center.cpu().tolist(),'original_energy':e9.cpu().tolist(),
                   'folded_energy':w.square().sum(1).cpu().tolist(),
                   'rule':'preserve DC; reduce historical spatial noise energy to original nine-frame energy; current frame unchanged'}

def run(kind,out):
    scene='night_'+kind;source=night_source(kind)
    config=json.loads((V13/'night_calibration_final'/scene/'manifest.json').read_text())['config']
    from ir_sr.training import dataset_for_config
    labels=[('control9',9,False,False),('fp16_9',9,True,False),('fp16_5',5,True,False),
            ('fp16_7',7,True,False),('noise5',5,True,True),('noise7',7,True,True)]
    boards={};counts={};details={}
    for name,count,half,filtered in labels:
        board=make_board(source,count,torch.zeros(1,4).cuda(),kind)
        if filtered:
            front,record=noise_filter(source.front.temporal_conv,board.source.front.temporal_conv)
            board.source.front.temporal_conv=front
            details[name]=record
        if half:board.source.front.half()
        board.requires_grad_(False)
        boards[name]=board;counts[name]=count
    val=dataset_for_config(config,'val')
    models={n:Adapter(b,list(range(9-counts[n],9))) for n,b in boards.items()}
    result=validation(models,config,val,[scene],lambda r:torch.zeros(1,4))
    eligible={n:permitted(m,result[scene]['control9']) for n,m in result[scene].items()}
    (out/'validation.json').write_text(json.dumps({'metrics':result,'eligible':eligible,'filters':details},indent=2))
    print('NIGHT_PRECISION_VALIDATION',kind,json.dumps(eligible),flush=True)
    del models
    data=dataset_for_config(config,'test');row=next(r for r in data.records if r['scene_id']==scene)
    from ir_sr.sequence_normalization import regional_key
    nframes=int(data.base.sequence_normalization.by_key[regional_key(row)]['frames'])
    metric=GapMetrics(row['eval_crop_tlhw'],boards)
    metric.prepare_background([read_gt(config,row,f) for f in range(nframes)])
    with torch.inference_mode():
        for frame in range(nframes):
            rr=dict(row,frame_id=frame)
            raw=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(rr,crop_tlhw=(0,0,1024,1280));context=context[None].cuda()
            ps={n:b(raw[:,-counts[n]:].contiguous(),context)[0,0,::3,::3].float().cpu().numpy() for n,b in boards.items()}
            metric.update(frame,read_gt(config,row,frame),ps)
    prepared=[]
    for p in sorted((V13/'night_calibration_final'/scene).glob('test_*.npz')):
        with np.load(p,allow_pickle=False) as a:
            raw=torch.from_numpy(a['nine_raw']).cuda()
            context=torch.from_numpy(a['reference_thumb']).cuda()
        if kind=='ordinary':raw=raw.half()
        prepared.append((raw,context))
    timing={}
    for name,board in boards.items():
        timing[name]=measure(board,[(x[:,-counts[name]:].contiguous(),c) for x,c in prepared],50,300)
    report=metric.report()
    report.update(scene=scene,validation_eligible=eligible,actual_input_channels=counts,
        filters=details,timing=timing,test_used_for_selection=False,automatically_adopted=False,NPU_measured=False)
    (out/'full_field_results.json').write_text(json.dumps(report,indent=2));save_gallery(metric,out)
    summary={k:report[k] for k in ['scene','summary','target_observations','GT_tracks','validation_eligible','actual_input_channels','filters']}
    summary['timing']={n:{k:{j:v for j,v in values.items() if j in ['mean_ms','p95_ms']} for k,values in t.items()} for n,t in timing.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('NIGHT_PRECISION_COMPLETE',kind,json.dumps({n:t['full_synchronized_wall']['mean_ms'] for n,t in timing.items()}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['ordinary','special'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.scene,a.out)
if __name__=='__main__':main()

