import argparse
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
MOTION=ROOT/'runs/SS928-MOTION-SPEED-20260930'
sys.path[:0]=[str(V13/'runtime'),str(MOTION/'runtime')]
from run_compact import setup,GROUPS,sha
from compact_model import BoardCompact
from equivalent_model import BoardHalfOptimized
from integer_output import IntegerOutput,ByteReorder
from probe_integer_output import ByteAfterRows
from finalize_candidates import benchmark
from prepare_deployment import export_model,ONNX_SITE
from speed_variants import RepeatByteReorder,replace_samples,NightFixed,ConvSample,NightByte
from baseline_contract import load_group,independent_control


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--night',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--reference-only',action='store_true');p.add_argument('--tail-only',action='store_true');p.add_argument('--quarter-only',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=a.resume);sys.path.insert(0,str(ONNX_SITE))
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'scenes':{}}
    if a.resume and (a.out/'all_speed.json').exists():report=json.loads((a.out/'all_speed.json').read_text())
    if a.night:
        os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
        sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src')]
        from reference_layout import load_candidate as load_ref
        from lowres_reference import load_candidate as load_low
        for scene in ['ordinary','special']:
            source=load_ref(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else load_low(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
            source=source.eval();directory=V13/'night_calibration_final'/('night_'+scene)
            sample=np.load(directory/'test_00.npz');args=tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw','reference_thumb'])
            repeat=NightFixed(source,scene).cuda().eval()
            fused=NightFixed(source,scene,fuse=True).cuda().eval()
            conv=copy.deepcopy(repeat);conv.sample=ConvSample(12 if scene=='ordinary' else 4,512,640).cuda()
            variants={'original_byte':ByteAfterRows(source),'legacy_repeat':NightByte(source).cuda(),'repeat_byte':repeat,'half_reference':replace_samples(copy.deepcopy(repeat),'half'),
                      'conv_reference':conv,'fused_repeat':fused,'fused_half_reference':replace_samples(copy.deepcopy(fused),'half')}
            if a.tail_only:
                from linear_tail_corrections import CorrectedLinearTail
                corrected=copy.deepcopy(repeat);corrected.fused=CorrectedLinearTail(source.core.model.tail,source.output.conv).cuda()
                variants={'original_byte':variants['original_byte'],'repeat_byte':repeat,'fused_corrected':corrected}
            measure('night_'+scene,variants,args,directory,report,a.out)
    else:
        for group in ['day','light','heavy']:
            compact,base,config,_,base_path,df,box_for=load_group(group)
            report.setdefault('independent_baselines',{})[group]=independent_control(group,compact)
            data=df(config,'test')
            for scene in GROUPS[group]['scenes']:
                row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda()
                directory=V13/'deployment'/scene
                if not (directory/'test_00.npz').exists():directory=V13/'deployment_corrected'/scene
                sample=np.load(directory/'test_00.npz')
                args=tuple(torch.from_numpy(sample[k]).cuda().half() for k in ['nine_raw','reference_thumb'])
                for kind,source,factor in [('half',BoardHalfOptimized(base,box),2)]+([] if group=='day' else [('quarter',BoardCompact(compact,box),4)]):
                    if a.reference_only and (kind!='quarter' or scene=='weather_heavy_c32'):continue
                    if a.quarter_only and kind!='quarter':continue
                    if scene+'_'+kind in report['scenes']:continue
                    source=source.cuda().half().eval();integer=IntegerOutput(source,factor).cuda().eval()
                    repeat=copy.deepcopy(integer);repeat.output=RepeatByteReorder(factor).cuda()
                    variants={'original_byte':ByteAfterRows(source),'integer_byte':integer,'repeat_byte':repeat,
                              'half_reference':replace_samples(copy.deepcopy(repeat),'half'),
                              'conv_reference':replace_samples(copy.deepcopy(repeat),'conv')}
                    if a.reference_only:variants={k:variants[k] for k in ['original_byte','repeat_byte','conv_reference']}
                    measure(scene+'_'+kind,variants,args,directory,report,a.out)
    print('ALL_SPEED_COMPLETE',flush=True)


def measure(name,variants,args,directory,report,out):
    r={'cases':{},'source_samples':str(directory),'input_dtypes':[str(v.dtype) for v in args]}; reference=variants['original_byte']
    with torch.inference_mode():expected=reference(*args)
    for label,m in variants.items():
        with torch.inference_mode():
            actual=m(*args);delta=(actual.float()-expected.float()).abs()
            control={'mae_gray':float(delta.mean()),'max_gray':float(delta.max()),'equal_pixels':bool(torch.equal(actual,expected))}
        if label=='legacy_repeat' or (label in ['integer_byte','repeat_byte'] and not name.startswith('night')):assert control['equal_pixels'],(name,label,control)
        timing=benchmark(m,args)
        if name.startswith('night'):timing['precision']='original mixed input/weight types preserved'
        r['cases'][label]={'control_first_frame':control,'GPU':timing,
                          'fixed_conv_samples':sum(isinstance(v,ConvSample) for v in m.modules())}
        print('ALL_SPEED',name,label,timing['mean_ms'],json.dumps(control),flush=True)
        if label!='original_byte':
            path=out/(name+'_'+label+'.onnx')
            if label=='legacy_repeat':
                import onnx
                torch.onnx.export(m,args,str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
                graph=onnx.load(str(path));onnx.checker.check_model(graph);ops=sorted({v.op_type for v in graph.graph.node});assert 'GridSample' not in ops
                metadata={'path':str(path),'sha256':sha(path),'operators':ops,'retains_original_night_Resize':True}
            else:metadata=export_model(m,args,path,['nine_raw','reference_thumb'],'display_gray')
            r['cases'][label]['graph']=metadata
        torch._dynamo.reset();torch.cuda.empty_cache()
    report['scenes'][name]=r;(out/'all_speed.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
