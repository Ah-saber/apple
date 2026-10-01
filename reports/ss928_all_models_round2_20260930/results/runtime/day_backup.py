"""Preserve the deployed current-only day alternative; keep nine-frame input boundary."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930';MOTION=ROOT/'runs/SS928-MOTION-SPEED-20260930'
sys.path[:0]=[str(V13/'runtime'),str(MOTION/'runtime')]
from run_compact import setup,OLD,sha
from equivalent_model import BoardHalfOptimized
from integer_output import IntegerOutput
from speed_variants import RepeatByteReorder
from probe_integer_output import ByteAfterRows
from finalize_candidates import benchmark
from prepare_deployment import ONNX_SITE,export_model


class FoldedNine(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,c):return self.model(x[:,-1:],c)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    _,primary,config,_,_,df,box_for=setup('day');sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    from half_student import make_student
    from train_quarter_student import BoxQuarter
    from ir_sr.model import GlobalReference
    path=V13/'release/models/baseline/half-day-rawskip.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    ref=GlobalReference(16,config.get('absolute_raw_reference',False),config.get('reference_pyramid',False),config.get('reference_sensor_y',False))
    base=BoxQuarter(make_student(ref,state),current_only=state.get('current_only',False));base.load_state_dict(state['model']);assert base.current_only
    data=df(config,'test');row=next(r for r in data.records if r['scene_id']=='day_normal');box=box_for(row).cuda()
    original=BoardHalfOptimized(base,box).cuda().float().eval();folded=BoardHalfOptimized(base,box,single_frame=True).cuda().float().eval()
    controls=[]
    with torch.inference_mode():
        for frame in [0,8,60,119]:
            rr=dict(row,frame_id=frame);x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr);c=c[None].cuda()
            expected=original(x,c);actual=folded(x[:,-1:],c);err=(actual-expected).abs()
            record={'frame':frame,'mean_gray':float(err.mean()),'max_gray':float(err.max())};assert record['max_gray']<.01;controls.append(record)
    source=original.half().eval();integer=IntegerOutput(source,2).cuda();repeat=copy.deepcopy(integer);repeat.output=RepeatByteReorder(2).cuda()
    floating=copy.deepcopy(integer);floating.output=RepeatByteReorder(2,to_byte=False).cuda()
    folded_integer=IntegerOutput(folded.half().eval(),2).cuda();folded_integer.output=RepeatByteReorder(2).cuda()
    cases={'original_float':source,'original_byte':ByteAfterRows(source),'integer':integer,'repeat':repeat,'float_repeat':floating,'folded_repeat':FoldedNine(folded_integer)}
    exact={k:0 for k in ['integer','repeat','float_repeat']};numeric=[]
    with torch.inference_mode():
        for frame in range(120):
            rr=dict(row,frame_id=frame);x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda().half();c,_=data.context_for(rr);c=c[None].cuda().half()
            original_float=source(x,c);expected=original_float.byte()
            for name in exact:
                actual=cases[name](x,c);assert torch.equal(actual,original_float if name=='float_repeat' else expected),(frame,name);exact[name]+=1
            err=(cases['folded_repeat'](x,c).float()-expected.float()).abs();numeric.append({'frame':frame,'mae_gray':float(err.mean()),'max_gray':float(err.max())})
            if frame%30==0:print('DAY_BACKUP',frame,flush=True)
    report={'NPU_measured':False,'weights':str(path),'weights_sha256':sha(path),'current_only_was_already_trained':True,'external_input_still_nine':True,
        'FP32_algebra_controls':controls,'same_backend_layout_exact_frames':exact,'FP16_fold_numeric_controls':numeric,'cases':{},'packaged':False,'pushed':False}
    for name,model in cases.items():
        model.requires_grad_(False)
        x,c=x.clone(),c.clone()
        record={'GPU':benchmark(model,(x,c))}
        if name!='original_byte':record['graph']=export_model(model,(x,c),a.out/(name+'.onnx'),['nine_raw','reference_thumb'],'display_gray')
        report['cases'][name]=record;torch._dynamo.reset();torch.cuda.empty_cache();print('DAY_BACKUP_SPEED',name,record['GPU']['mean_ms'],flush=True)
    (a.out/'day_backup.json').write_text(json.dumps(report,indent=2));print('DAY_BACKUP_COMPLETE',flush=True)


if __name__=='__main__':main()
