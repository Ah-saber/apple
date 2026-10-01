import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
from run_compact import setup,GROUPS,sha,score
from adaptive_history import AdaptiveHistory,HistorySignal
from baseline_contract import load_group,independent_control


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--day-only',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'trained_new_weights':False,'uses_only_current_and_past_inputs':True,'groups':{}}
    for group in (['day'] if a.day_only else ['day','light','heavy']):
        compact,base,config,_,basepath,df,box_for=load_group(group)
        baseline_control=independent_control(group,compact)
        if group!='day':compact.load_state_dict(torch.load(V13/(group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)['model'])
        cache=ROOT/'runs'/('SS928-FIVE-SCENE-'+GROUPS[group]['cache']+'-20260929')/'train_cache.pt'
        saved=torch.load(cache,map_location='cpu',weights_only=False);data=saved['values'];x=data['stack'][:,1].float()
        delta=(data['gt'][:,1]-data['gt'][:,0])*255
        if group=='day':delta=delta-delta.flatten(1).median(1).values[:,None,None,None]
        delta=delta.abs()
        static=F.max_pool2d((delta>1).float(),7,1,3)==0
        signal=HistorySignal(1,1).signal(x);selected=signal[static]
        assert selected.numel()>100
        thresholds={str(p):float(torch.quantile(selected,p/100)) for p in [80,90,98]}
        calibration={'static_pixels':selected.numel(),'zero_fraction':float((selected<=1e-5).float().mean()),'positive_conditioning':False,'training_GT_scalar_drift_removed':group=='day'}
        if thresholds['98']<=1e-5:
            positive=selected[selected>1e-5]
            if positive.numel()>100:
                thresholds={str(p):float(torch.quantile(positive,p/100)) for p in [80,90,98]}
                calibration.update(positive_conditioning=True,positive_pixels=positive.numel(),reason='unconditional static percentiles are zero; use measured nonzero static residuals')
            else:
                thresholds={str(p):float(torch.quantile(signal.flatten(),p/100)) for p in [80,90,98]}
                assert thresholds['80']>1e-5,('training RAW also degenerate',group)
                calibration.update(fallback='all_training_RAW_pixels',reason='GT static filter retained only repeated beginning frames; full training RAW percentiles include moving regions and are conservative signal thresholds, not measured noise standard deviation')
        g={'calibration_cache':str(cache),'calibration_sha256':sha(cache),'calibration_audit':calibration,'input_unit_thresholds':thresholds,'kinds':{},'independent_baseline_control':baseline_control}
        for kind,source in [('half',base)]+([] if group=='day' else [('quarter',compact)]):
            source=source.cuda().float().eval();val=df(config,'val')
            baseline=score(source,val,config,GROUPS[group]['scenes'],box_for)
            controls=[]
            with torch.inference_mode():
                xx=x[:2].cuda();cc=data['context'][:2,1].cuda();bb=data['box'][:2,1].cuda()
                for force in [0.,1.]:
                    model=AdaptiveHistory(source,kind,1,force).cuda().float().eval()
                    actual=model.native(xx,cc,bb)
                    expected=source.native(xx if force==0 else xx[:,-1:].expand_as(xx),cc,bb)
                    error=(actual-expected).abs()*255
                    control={'force':force,'mae_gray':float(error.mean()),'max_gray':float(error.max())}
                    assert control['max_gray']<.01,control;controls.append(control)
                with torch.autocast('cuda',dtype=torch.bfloat16):bf16=source.native(xx,cc,bb).float()
                fp32=source.native(xx,cc,bb);error=(bf16-fp32).abs()*255
                precision={'BF16_vs_FP32_crop_mean_gray':float(error.mean()),'max_gray':float(error.max())}
            trials={}
            for percentile,tau in thresholds.items():
                model=AdaptiveHistory(source,kind,tau).cuda().float().eval()
                trials[percentile]=score(model,val,config,GROUPS[group]['scenes'],box_for)
            baseline_mean=float(np.mean([v['psnr_db'] for v in baseline['summary'].values()]))
            # Conservative validation guard: permit at most 0.15 dB mean loss.
            eligible=[p for p in thresholds if np.mean([v['psnr_db'] for v in trials[p]['summary'].values()])>=baseline_mean-.15]
            chosen=min(eligible,key=int) if eligible else '98'
            g['kinds'][kind]={'baseline_val':baseline,'validation_trials':trials,'chosen_percentile':chosen,
                'chosen_threshold':thresholds[chosen],'passed_mean_val_guard':bool(eligible),'endpoint_controls':controls,'precision_probe':precision}
            print('ADAPTIVE_PROBE',group,kind,chosen,'guard',bool(eligible),'precision',precision,flush=True)
        report['groups'][group]=g;(a.out/'adaptive_probe.json').write_text(json.dumps(report,indent=2))
    print('ADAPTIVE_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
