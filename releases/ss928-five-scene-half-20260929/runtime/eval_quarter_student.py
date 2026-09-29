"""Frozen-test quality and complete-output GPU time for a quarter student."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--night-runtime',type=Path,required=True)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--speed',action='store_true')
    p.add_argument('--detail-scale',type=float,default=1.0)
    a=p.parse_args()
    sys.path[:0]=[str(a.code/'src'),str(a.night_runtime),
                  str(Path(__file__).parent)]
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from half_student import make_student
    from train_quarter_student import BoxQuarter,box_for
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    initial=torch.load(a.config_checkpoint,map_location='cpu',weights_only=False)
    config=initial['config']
    reference_state=(torch.load(a.reference_checkpoint,map_location='cpu',
                    weights_only=False) if a.reference_checkpoint else initial)
    source=inference_model(reference_state['config'],reference_state['model'])
    reference=copy.deepcopy(source.global_reference)
    del source
    checkpoint=torch.load(a.student_checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('format')=='quarter_detail_refiner_v1':
        from refine_quarter_detail import DetailRefiner
        base_state=torch.load(checkpoint['base_checkpoint'],map_location='cpu',weights_only=False)
        base=BoxQuarter(make_student(reference,base_state),
            current_only=base_state.get('current_only',False))
        model=DetailRefiner(base)
        model.scale=a.detail_scale
    else:
        model=BoxQuarter(make_student(reference,checkpoint),
            current_only=checkpoint.get('current_only',False))
    model.load_state_dict(checkpoint['model'],strict=True)
    model=model.cuda().eval()
    data=dataset_for_config(config,'test')
    scenes=checkpoint.get('scenes',config['scene_ids'])
    results={scene:[] for scene in scenes}
    sample=None
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for row in data.records:
            scene=row['scene_id']
            if scene not in results:
                continue
            stack=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(row)
            context=context[None].cuda()
            box=box_for(row).cuda()
            output=model.native(stack,context,box).float().clamp(0,1)
            target=np.asarray(Image.open(Path(config['data_root'])/
                                         row['target']['path']),dtype=np.float32)/255.
            y,x,h,w=row['eval_crop_tlhw']
            y,x,h,w=y+3,x+3,h-6,w-6
            delta=(output[0,0,y:y+h,x:x+w]-torch.from_numpy(
                target[y:y+h,x:x+w]).cuda())*255
            mse=float(delta.square().mean())
            results[scene].append({'sample_id':row['sample_id'],
                'psnr_db':float(10*np.log10(255**2/max(mse,1e-12))),
                'mae_gray':float(delta.abs().mean()),
                'bias_gray':float(delta.mean())})
            if sample is None:sample=(stack,context,box)
    summary={key:{'samples':len(values),
                  'psnr_mean_db':float(np.mean([v['psnr_db'] for v in values])),
                  'mae_mean_gray':float(np.mean([v['mae_gray'] for v in values]))}
             for key,values in results.items()}
    report={'student_checkpoint':str(a.student_checkpoint),
            'config_checkpoint':str(a.config_checkpoint),
            'val_selection':checkpoint.get('val_psnr_db'),
            'test':summary,'samples':results}
    if a.speed:
        model=model.half().to(memory_format=torch.channels_last)
        stack,context,box=sample
        torch._inductor.config.triton.cudagraphs=False
        compiled=torch.compile(model,fullgraph=True,
                               options={'triton.cudagraphs':False})
        with torch.inference_mode():
            for _ in range(30):output=compiled(stack,context,box)
            torch.cuda.synchronize()
            times=[]
            for _ in range(300):
                start=torch.cuda.Event(enable_timing=True)
                stop=torch.cuda.Event(enable_timing=True)
                start.record()
                output=compiled(stack,context,box)
                stop.record();stop.synchronize()
                times.append(start.elapsed_time(stop))
        assert tuple(output.shape)==(1,1,3072,3840)
        report['server_gpu_compiled_fp16']={
            'mean_ms':float(np.mean(times)),
            'p95_ms':float(np.percentile(times,95)),
            'n':len(times),'input_shape':list(stack.shape),
            'output_shape':list(output.shape),
            'cuda_graphs':False,'TF32':False}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'test':summary,
                      'speed':report.get('server_gpu_compiled_fp16')},
                     indent=2),flush=True)


if __name__=='__main__':main()
