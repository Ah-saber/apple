"""Measure full-image component GPU times of the current fast nine-frame model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def time_step(fn):
    for _ in range(20):
        value = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return value, {'mean_ms': float(np.mean(samples)),
                   'p95_ms': float(np.percentile(samples, 95))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.code/'src'), str(a.runtime), str(Path(__file__).parent)]
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from ir_sr.sequence_normalization import allowed_region
    from fast_nine_inference import build_fast_model
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    f = build_fast_model(state['config'], state['model'], inference_model).cuda().eval()
    ds = dataset_for_config(state['config'], 'val')
    row = ds.records[0]
    x = ds.normalized_stack(row, (0,0,1024,1280))[None].cuda()
    c,_ = ds.context_for(row)
    c = c[None].cuda()
    rt,rl,rh,rw = allowed_region(row)
    box = torch.tensor([[-rt/rh,-rl/rw,(1024-rt)/rh,(1280-rl)/rw]],
                       dtype=torch.float32,device='cuda')
    m=f.model
    with torch.inference_mode():
        trajectory, t1 = time_step(lambda: m._trajectory_features(x))
        gate_input=torch.cat((x,trajectory),1)
        joined, t2 = time_step(lambda: f.second(F.relu(f.first(f.half_input(gate_input)))).float())
        denoised=x[:,-1:].float()+joined[:,:1]*(1.-torch.sigmoid(joined[:,1:]+0.5))
        features,t3 = time_step(lambda: m.body(m.head(m.down(f.half_input(denoised)))))
        reference,t4 = time_step(lambda: m.global_reference(f.half_input(c),box,features.shape[-2:]))
        packed,t5 = time_step(lambda: m.upsample[0](m.tail(features+reference.to(features.dtype))))
        _,t6 = time_step(lambda: F.pixel_shuffle(packed.float(),6))
        _,total = time_step(lambda: f(x,c,box))
    result={'trajectory':t1,'motion_correction':t2,'body':t3,'reference':t4,
            'tail_output_conv':t5,'pixel_shuffle':t6,'whole_eager':total}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
