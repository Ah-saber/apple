"""Run exact complete ONNX graphs and preserve layer references for real frames."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_compact import ROOT, OLD, setup, GROUPS, sha
from prepare_deployment import ONNX_SITE, arrays
from equivalent_model import BoardHalfOptimized
from compact_model import BoardCompact


def delta(actual, expected):
    d=actual.astype(np.float32)-expected.astype(np.float32)
    return {'mae_gray':float(np.abs(d).mean()),'max_gray':float(np.abs(d).max()),
            'bias_gray':float(d.mean()),'shape':list(actual.shape)}


def ort_run(graph, sample):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    feed={}
    for v in session.get_inputs():
        x=sample['nine_raw' if v.name=='current_raw' else v.name]
        if v.name=='current_raw':x=x[:,-1:]
        feed[v.name]=np.ascontiguousarray(x,dtype={'tensor(float)':np.float32,'tensor(float16)':np.float16}[v.type])
    result=session.run(None,feed)[0]
    assert tuple(result.shape)==(1,1,3072,3840)
    # All these complete graphs explicitly replicate each native pixel three times.
    return result[:,:,::3,::3].astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    dest=a.root/'verification';dest.mkdir(exist_ok=True)
    report={'onnxruntime_provider':'CPUExecutionProvider','NPU_measured':False,'cases':{},'layers':{}}
    with torch.inference_mode():
        for group in ['day','light','heavy']:
            _,base,config,teacher,base_path,df,box_for=setup(group)
            base=base.cuda().float().eval()
            for scene in GROUPS[group]['scenes']:
                directory=a.root/'deployment_corrected'/scene
                manifest=json.loads((directory/'manifest.json').read_text())
                sample=np.load(directory/'test_00.npz',allow_pickle=False)
                x=torch.from_numpy(sample['nine_raw']).cuda();c=torch.from_numpy(sample['reference_thumb']).cuda()
                box=torch.tensor([manifest['context_box']],device='cuda')
                model=BoardHalfOptimized(base,box).cuda().float().eval()
                for precision in ['fp32','fp16']:
                    m=copy.deepcopy(model).to(dtype=torch.float32 if precision=='fp32' else torch.float16)
                    dtype=m.base.quarter.front.first.weight.dtype
                    stages=m.stages(x.to(dtype),c.to(dtype))
                    path=directory/f'pc_layers_detailed_{precision}.npz'
                    np.savez_compressed(path,**arrays(stages))
                    report['layers'][scene+'_'+precision]={'path':str(path),'sha256':sha(path),
                        'sample':str(directory/'test_00.npz'),'saved_as':'float32',
                        'computation_precision':precision,'stages':{k:list(v.shape) for k,v in stages.items()}}
                    expected=(F.pixel_shuffle(stages['preclip_phases'].clamp(0,1)*255,2)).float().cpu().numpy()
                    if precision=='fp32':
                        check=delta(expected,sample['optimized_native_gray']);assert check['max_gray']<.01,check
                    graph=directory/f'{scene}_equivalent_{precision}_rows32.onnx'
                    actual=ort_run(graph,sample)
                    record=delta(actual,expected)
                    record.update({'graph':str(graph),'sha256':sha(graph),'sample':str(directory/'test_00.npz'),
                                   'comparison':'exact ONNX CPU against same precision PyTorch GPU'})
                    if precision=='fp32': assert record['max_gray']<.01,record
                    report['cases'][scene+'_'+precision]=record
                    print('ORT',scene,precision,record,flush=True)
                    del m,stages
                if group=='day':
                    from half_student import make_student
                    from train_quarter_student import BoxQuarter
                    state=torch.load(ROOT/'runs/SS928-FIVE-SCENE-HALF-DAY-RAWSKIP-20260929/best.pt',map_location='cpu',weights_only=False)
                    alt=BoxQuarter(make_student(copy.deepcopy(base.quarter.reference),state),current_only=True)
                    alt.load_state_dict(state['model']);alt=alt.cuda().float().eval()
                    expected=(alt.native(x,c,box).clamp(0,1)*255).cpu().numpy()
                    graph=directory/'day_current_folded_fp32_rows32.onnx'
                    record=delta(ort_run(graph,sample),expected)
                    assert record['max_gray']<.01,record
                    record.update({'graph':str(graph),'sha256':sha(graph),'comparison':'folded current-only graph against original current-only source'})
                    report['cases']['day_current_folded_fp32']=record
            if group in ['light','heavy']:
                for suffix in ['fused','temporal']:
                    run=a.root/(group+'_'+suffix)
                    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
                    GROUPS[group].update(ck['spec'])
                    model,*_=setup(group);model.load_state_dict(ck['model']);model=model.cuda().float().eval()
                    for scene in GROUPS[group]['scenes']:
                        directory=a.root/'deployment_corrected'/scene
                        sample=np.load(directory/'test_00.npz',allow_pickle=False)
                        manifest=json.loads((directory/'manifest.json').read_text())
                        box=torch.tensor([manifest['context_box']],device='cuda')
                        x=torch.from_numpy(sample['nine_raw']).cuda();c=torch.from_numpy(sample['reference_thumb']).cuda()
                        board=BoardCompact(model,box).cuda().half().eval()
                        expected=board(x.half(),c.half()).float().cpu().numpy()[:,:,::3,::3]
                        graph=run/'onnx'/scene/f'{scene}_compact_fp16_rows32.onnx'
                        record=delta(ort_run(graph,sample),expected)
                        record.update({'graph':str(graph),'sha256':sha(graph),'comparison':'exact compact FP16 ONNX CPU against PyTorch GPU'})
                        report['cases'][scene+'_'+suffix+'_compact_fp16']=record
                        print('ORT_COMPACT',scene,suffix,record,flush=True)
                        del board
            (dest/'exact_graph_verification.json').write_text(json.dumps(report,indent=2))
    print('EXACT_GRAPH_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
