"""Explicitly changed current-only controls; no automatic release or NPU claims."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup, GROUPS, OLD, sha
from equivalent_model import BoardHalfOptimized
from compact_model import BoardCompact
from finalize_candidates import benchmark
from prepare_deployment import export_model, ONNX_SITE


def fold_compact(source):
    result=copy.deepcopy(source)
    old=result.front
    conv=nn.Conv2d(1,old.out_channels,old.kernel_size,old.stride,old.padding)
    with torch.no_grad():
        conv.weight.copy_(old.weight.sum(1,keepdim=True));conv.bias.copy_(old.bias)
    result.front=conv.to(old.weight)
    result.raw_kernel=result.raw_kernel.sum(1,keepdim=True)
    return result


class OutputByte(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,c):return self.model(x,c).to(torch.uint8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'semantic_change':'current-only main input; not equivalent to original temporal inference',
            'weights_retrained':False,'packaged':False,'pushed':False,'scenes':{}}
    for group,scene in [('light','weather_light'),('heavy','weather_heavy')]:
        ck=torch.load(PREVIOUS/(group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)
        GROUPS[group].update(ck['spec'])
        compact,base,config,teacher,base_path,df,box_for=setup(group)
        compact.load_state_dict(ck['model']);compact=compact.cuda().float().eval();base=base.cuda().float().eval()
        data=df(config,'test');row=next(r for r in data.records if r['scene_id']==scene)
        x=data.normalized_stack(row,(0,0,1024,1280))[None].cuda();c,_=data.context_for(row);c=c[None].cuda();box=box_for(row).cuda()
        forced=copy.deepcopy(base);forced.current_only=True
        current_half=BoardHalfOptimized(forced,box,single_frame=True).cuda().float().eval()
        current_quarter=BoardCompact(fold_compact(compact),box).cuda().float().eval()
        cases={'half_nine':BoardHalfOptimized(base,box).cuda().float().eval(),
               'half_current':current_half,
               'quarter_nine':BoardCompact(compact,box).cuda().float().eval(),
               'quarter_current':current_quarter,
               'quarter_current_byte':OutputByte(copy.deepcopy(current_quarter))}
        controls=[]
        with torch.inference_mode():
            for frame in [8,60,119]:
                rr=dict(row,frame_id=frame)
                xx=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();cc,_=data.context_for(rr);cc=cc[None].cuda()
                for name,model,source in [('half',current_half,base),('quarter',current_quarter,compact)]:
                    expected=source.native(xx[:,-1:].expand_as(xx),cc,box).clamp(0,1)*255
                    actual=model(xx[:,-1:],cc)[:,:,::3,::3]
                    d=(actual-expected).abs()
                    record={'frame':frame,'model':name,'mae_gray':float(d.mean()),'max_gray':float(d.max())}
                    assert record['max_gray']<.01,record;controls.append(record)
        record={'algebraic_controls_against_repeated_current':controls,'cases':{},'graphs':[]}
        for name,m in cases.items():
            m=copy.deepcopy(m).half().eval()
            single='current' in name
            args=(x[:,-1:].half() if single else x.half(),c.half())
            record['cases'][name]=benchmark(m,args)
            record['cases'][name]['input_channels']=1 if single else 9
            print('TIMING',scene,name,record['cases'][name]['mean_ms'],flush=True)
            if single:
                path=a.out/f'{scene}_{name}_fp16.onnx'
                record['graphs'].append(export_model(m,args,path,['current_raw','reference_thumb'],'display_gray'))
            torch._dynamo.reset();torch.cuda.empty_cache()
        report['scenes'][scene]=record
        (a.out/'current_fold_gpu.json').write_text(json.dumps(report,indent=2))
    print('CURRENT_FOLD_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
