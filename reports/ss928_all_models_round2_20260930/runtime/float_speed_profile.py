"""Full forward timing with original float output contract and convolution accounting."""
import argparse,copy,json,os,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path[:0]=[str(V13/'runtime'),str(ROOT/'runs/SS928-MOTION-SPEED-20260930/runtime')]
from baseline_contract import load_group
from run_compact import GROUPS
from equivalent_model import BoardHalfOptimized
from compact_model import BoardCompact
from integer_output import IntegerOutput
from speed_variants import RepeatByteReorder,NightByte
from finalize_candidates import benchmark

def profile(model,args):
    rows=[];handles=[]
    for name,m in model.named_modules():
        if not isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):continue
        def hook(module,inputs,output,name=name):
            n,ci,hi,wi=inputs[0].shape;n,co,ho,wo=output.shape;k=module.kernel_size
            mac=n*(hi*wi if isinstance(module,nn.ConvTranspose2d) else ho*wo)*ci*co*k[0]*k[1]//module.groups
            rows.append({'name':name,'operator':type(module).__name__,'MAC':int(mac),'learned_parameters':any(p.requires_grad for p in module.parameters()),'output_bytes':output.numel()*output.element_size(),'output_shape':list(output.shape)})
        handles.append(m.register_forward_hook(hook))
    with torch.inference_mode():model(*args)
    for h in handles:h.remove()
    return {'convolution_MAC':sum(v['MAC'] for v in rows),'largest_convolution_output_bytes':max(v['output_bytes'] for v in rows),'layers':rows,'limits':'convolution arithmetic accounting, padded positions included; excludes resize/layout memory and is not NPU latency or peak memory'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'scenes':{}}
    def run(name,source,repeat,args):
        with torch.inference_mode():assert torch.equal(source(*args),repeat(*args)),name
        r={'same_backend_first_frame_exact':True,'profile':profile(source,args),'cases':{}}
        for label,m in [('original_float',source),('repeat_float',repeat)]:
            timing=benchmark(m,args);timing['precision']='original weights and input dtypes preserved'
            with torch.inference_mode():
                eager=m(*args);compiled=torch.compile(m,mode='default')(*args)
                delta=(eager.float()-compiled.float()).abs()
            r['cases'][label]={'GPU':timing,'compiled_vs_eager_first_frame':{'mean_gray':float(delta.mean()),'max_gray':float(delta.max()),'output_dtype':str(compiled.dtype),'input_dtypes':[str(v.dtype) for v in args]}}
            torch._dynamo.reset();torch.cuda.empty_cache()
        report['scenes'][name]=r;(a.out/'float_speed_profile.json').write_text(json.dumps(report,indent=2));print('FLOAT_SPEED',name,{k:v['GPU']['mean_ms'] for k,v in r['cases'].items()},flush=True)
    for group in ['day','light','heavy']:
        q,base,config,_,_,df,box_for=load_group(group);data=df(config,'test')
        for scene in GROUPS[group]['scenes']:
            row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda()
            directory=V13/'deployment'/scene
            if not (directory/'test_00.npz').exists():directory=V13/'deployment_corrected'/scene
            sample=np.load(directory/'test_00.npz');args=tuple(torch.from_numpy(sample[k]).cuda().half() for k in ['nine_raw','reference_thumb'])
            for kind,source,factor in [('half',BoardHalfOptimized(base,box),2)]+([] if group=='day' else [('quarter',BoardCompact(q,box),4)]):
                source=source.cuda().half().eval();repeat=IntegerOutput(source,factor).cuda().eval();repeat.output=RepeatByteReorder(factor,to_byte=False).cuda();run(scene+'_'+kind,source,repeat,args)
    os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
    sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src')]
    from reference_layout import load_candidate as refload
    from lowres_reference import load_candidate as lowload
    for scene in ['ordinary','special']:
        source=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
        directory=V13/'night_calibration_final'/('night_'+scene);sample=np.load(directory/'test_00.npz');args=tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw','reference_thumb'])
        repeat=NightByte(source).cuda().eval();repeat.source.output.output=RepeatByteReorder(2,to_byte=False).cuda();run('night_'+scene,source,repeat,args)
    print('FLOAT_SPEED_PROFILE_COMPLETE',flush=True)
if __name__=='__main__':main()
