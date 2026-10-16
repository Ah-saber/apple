"""GT-only validation windows with enough history to measure departures."""
import numpy as np
import torch
from gap_metrics_v5 import GapMetrics
import target_validation_rows_v2 as rows
from target_validation_v3 import cohorts
from run_reduced_v2 import read_gt

def history_frames(data,row):
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization
    name=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    frame=int(row['frame_id'])
    begin=next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin']<=frame<s['end'])
    return range(max(begin,frame-15),frame+1)

def validation(models,config,data,scenes,box_for):
    # Row selection remains GT-only. Only validation sequence/ROI is inspected.
    rows.history_frames=history_frames
    result={}
    with torch.inference_mode():
        for scene in scenes:
            selected,templates=rows.select_rows(data,config,scene);assert selected
            observations=[];scored=[]
            for row in selected:
                metric=GapMetrics(row['eval_crop_tlhw'],models)
                metric.background,metric.noise=templates[rows.key(row)]
                end=int(row['frame_id'])
                for f in history_frames(data,row):
                    rr=dict(row,frame_id=f)
                    raw=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
                    context,_=data.context_for(rr);context=context[None].cuda();box=box_for(rr).cuda()
                    predictions={name:(model.native(raw,context,box)[0,0].float().clamp(0,1)*255).cpu().numpy()
                                 for name,model in models.items()}
                    metric.update(f,read_gt(config,row,f),predictions)
                report=metric.report();observations.append(report)
                # Score only the last eight frames. Earlier frames establish tracks.
                scored.extend(r for r in report['frames'] if r['frame']>=max(0,end-7))
            records=[r for report in observations for r in report['targets']]
            grouped=cohorts(records,models);result[scene]={}
            for name in models:
                frames=[r['methods'][name] for r in scored]
                targets=[r['methods'][name] for r in records]
                keys=set().union(*(r.keys() for r in frames))
                result[scene][name]=dict(frames={k:float(np.mean([r[k] for r in frames if k in r])) for k in keys},
                    targets={k:float(np.mean([r[k] for r in targets])) for k in targets[0]} if targets else {},
                    target_observations=len(targets),cohorts=grouped[name],validation_history_frames=16,
                    frames_scored_per_window=8,GT_only_selection=True)
    return result
