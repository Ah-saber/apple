"""Prove byte-layout equality and time complete models, independently of training."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup,GROUPS,OLD
from compact_model import BoardCompact
from equivalent_model import BoardHalfOptimized
from finalize_candidates import benchmark
from prepare_deployment import ONNX_SITE,export_model
from integer_output import ByteReorder,IntegerOutput


class ByteAfterRows(torch.nn.Module):
    def __init__(self,source):super().__init__();self.source=source
    def forward(self,x,c):return self.source(x,c).to(torch.uint8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    import onnxruntime as ort
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    controls={}
    for factor in [2,4]:
        mapper=ByteReorder(factor).cuda()
        phases=torch.rand(1,factor**2,1024//factor,1280//factor,device='cuda').half()*255
        expected=F.interpolate(F.pixel_shuffle(phases,factor),scale_factor=3,mode='nearest').byte()
        actual=mapper(phases)
        assert torch.equal(actual,expected)
        path=a.out/f'phase{factor**2}_integer_output.onnx'
        graph=export_model(mapper,(phases,),path,['phases'],'display_gray')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        result=session.run(None,{'phases':phases.cpu().numpy()})[0]
        assert np.array_equal(result,expected.cpu().numpy())
        controls[str(factor)]={'torch_exact':True,'ORT_exact':True,'graph':graph}
    ck=torch.load(PREVIOUS/'light_temporal/best.pt',map_location='cpu',weights_only=False)
    GROUPS['light'].update(ck['spec']);compact,base,config,teacher,base_path,df,box_for=setup('light')
    compact.load_state_dict(ck['model']);compact=compact.cuda().float().eval();base=base.cuda().float().eval()
    data=df(config,'test');row=next(r for r in data.records if r['scene_id']=='weather_light')
    x=data.normalized_stack(row,(0,0,1024,1280))[None].cuda();c,_=data.context_for(row);c=c[None].cuda();box=box_for(row).cuda()
    report={'NPU_measured':False,'SDK_uint8_layout_support':'unmeasured','byte_rounding':'direct truncating Cast; actual SDK behavior must be checked',
            'layout_controls':controls,'models':{}}
    for name,source,factor in [('half',BoardHalfOptimized(base,box),2),('quarter',BoardCompact(compact,box),4)]:
        source=source.cuda().half().eval();integer=IntegerOutput(source,factor).cuda().eval()
        with torch.inference_mode():
            expected=source(x.half(),c.half()).byte();actual=integer(x.half(),c.half())
            assert torch.equal(actual,expected),name
        values={}
        for label,m in [('float_rows32',source),('byte_after_rows32',ByteAfterRows(copy.deepcopy(source))),('integer_byte',integer)]:
            values[label]=benchmark(m,(x.half(),c.half()))
            print('INTEGER_GPU',name,label,values[label]['mean_ms'],flush=True)
            torch._dynamo.reset();torch.cuda.empty_cache()
        graph=export_model(integer,(x.half(),c.half()),a.out/f'weather_light_{name}_integer_byte_fp16.onnx',['nine_raw','reference_thumb'],'display_gray')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        session=ort.InferenceSession(graph['path'],sess_options=options,providers=['CPUExecutionProvider'])
        output=session.run(None,{'nine_raw':x.half().cpu().numpy(),'reference_thumb':c.half().cpu().numpy()})[0]
        report['models'][name]={'GPU':values,'graph':graph,'same_backend_byte_layout_exact':True,
            'ORT_complete_shape':list(output.shape),'ORT_complete_dtype':str(output.dtype),
            'ORT_CPU_vs_PyTorch_GPU_byte_mae':float(np.abs(output.astype(np.float32)-expected.cpu().numpy()).mean()),
            'ORT_CPU_vs_PyTorch_GPU_byte_max':float(np.abs(output.astype(np.float32)-expected.cpu().numpy()).max())}
        (a.out/'integer_output_results.json').write_text(json.dumps(report,indent=2))
    print('INTEGER_OUTPUT_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
