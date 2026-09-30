"""Export evaluated candidates and measure complete models under one GPU protocol."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_compact import ROOT, OLD, GROUPS, setup, sha
from compact_model import BoardCompact
from equivalent_model import BoardHalfOptimized
from prepare_deployment import export_model, ONNX_SITE


def benchmark(model, inputs):
    model = torch.compile(model,mode='default')
    with torch.inference_mode():
        for _ in range(30): model(*inputs)
        torch.cuda.synchronize()
        times=[]
        for _ in range(300):
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); value=model(*inputs); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
        assert tuple(value.shape)==(1,1,3072,3840)
    return {'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
            'samples_ms':times,'warmup':30,'iterations':300,'shape':list(value.shape),
            'device':torch.cuda.get_device_name(0),'precision':'FP16','compile':'default',
            'cuda_graph':False,'TF32':False,'boundary':'prepared GPU input through complete display output'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--temporal',action='store_true')
    p.add_argument('--export-only',action='store_true'); a=p.parse_args()
    sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    from export_half_board import BoardHalf
    torch.set_num_threads(2); torch.manual_seed(930)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'export_only':a.export_only,'scenes':{}}
    for group in ['light','heavy']:
        out=a.root/(group+'_temporal' if a.temporal else group+'_fused')
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        GROUPS[group].update(ck['spec'])
        model,base,config,teacher,base_path,df,box_for=setup(group)
        model.load_state_dict(ck['model'],strict=True)
        model=model.cuda().float().eval(); base=base.cuda().float().eval()
        data=df(config,'test')
        for scene in GROUPS[group]['scenes']:
            row=next(r for r in data.records if r['scene_id']==scene)
            x=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            c,_=data.context_for(row); c=c[None].cuda(); box=box_for(row).cuda()
            board=BoardCompact(model,box).cuda().float().eval()
            with torch.inference_mode():
                source=model(x,c,box).float()
                d=(board(x,c).float()-source).abs()
                assert float(d.max())<.01,float(d.max())
                control=torch.rand(1,16,256,320,device='cuda')*255
                expect=F.interpolate(F.pixel_shuffle(control,4),scale_factor=3,mode='nearest')
                assert torch.equal(board.output(control),expect)
            export=out/'onnx'/scene; export.mkdir(parents=True,exist_ok=True)
            record={'checkpoint':str(out/'best.pt'),'checkpoint_sha256':sha(out/'best.pt'),
                    'source_to_frozen_mae_gray':float(d.mean()),'source_to_frozen_max_gray':float(d.max()),'graphs':[]}
            for precision in ['fp32','fp16']:
                m=copy.deepcopy(board).to(dtype=torch.float32 if precision=='fp32' else torch.float16)
                dtype=m.model.front.weight.dtype
                for layout in ['rows32','phases']:
                    m.layout=layout
                    record['graphs'].append(export_model(m,(x.to(dtype),c.to(dtype)),
                        export/f'{scene}_compact_{precision}_{layout}.onnx',
                        ['nine_raw','reference_thumb'],'display_gray' if layout!='phases' else 'sixteen_phases'))
            cases={'source_half':BoardHalf(copy.deepcopy(base),box,'rows32').cuda().half().eval(),
                   'equivalent_half':BoardHalfOptimized(base,box).cuda().half().eval(),
                   'compact_quarter':copy.deepcopy(board).half().eval()}
            for name,m in cases.items():
                if a.export_only: continue
                record[name+'_GPU']=benchmark(m,(x.half(),c.half()))
                print('GPU',scene,name,record[name+'_GPU']['mean_ms'],flush=True)
                torch._dynamo.reset(); torch.cuda.empty_cache()
            report['scenes'][scene]=record
            (a.root/('final_temporal.json' if a.temporal else 'final_fused.json')).write_text(json.dumps(report,indent=2))
            (export/'manifest.json').write_text(json.dumps(record,indent=2))
    print('FINALIZATION_COMPLETE',flush=True)


if __name__=='__main__':main()
