"""Find nine-frame trajectory cost and compare the previously verified fast statistic."""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def time_call(fn, warmup=10, repeats=40):
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            out = fn()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
    return out, {'median_ms': float(np.median(samples)),
                 'p95_ms': float(np.percentile(samples, 95)),
                 'min_ms': float(np.min(samples))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.code / 'src'), str(a.runtime)]
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config
    from trajectory_batched_stats import trajectory_features_batched_stats

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    model = inference_model(config, state['model']).cuda().eval()
    ds = dataset_for_config(config, 'val')
    row = ds.records[0]
    x = ds.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
    context, _ = ds.context_for(row)
    rt, rl, rh, rw = allowed_region(row)
    box = torch.tensor([[-rt/rh, -rl/rw, (1024-rt)/rh, (1280-rl)/rw]],
                       dtype=torch.float32, device='cuda')
    context = context[None].cuda()
    forward = lambda: model(x, context=context, context_box=box)
    result = {'checkpoint': str(a.checkpoint), 'scene': row['scene_id']}
    original = model._trajectory_features
    y_reference, result['original_forward'] = time_call(forward)
    _, result['original_trajectory'] = time_call(lambda: original(x))

    model._trajectory_features = types.MethodType(trajectory_features_batched_stats, model)
    y_fast, result['fast_forward'] = time_call(forward)
    _, result['fast_trajectory'] = time_call(lambda: model._trajectory_features(x))
    delta = (y_fast.float() - y_reference.float()).abs()
    result['fast_vs_original'] = {'mean_gray': float(delta.mean().item() * 255),
                                   'max_gray': float(delta.max().item() * 255)}

    zeros = torch.zeros((x.shape[0], 4, x.shape[-2], x.shape[-1]),
                        dtype=x.dtype, device=x.device)
    model._trajectory_features = lambda _x: zeros
    y_zero, result['zero_trajectory_forward'] = time_call(forward)
    delta_zero = (y_zero.float() - y_reference.float()).abs()
    result['zero_vs_original'] = {'mean_gray': float(delta_zero.mean().item() * 255),
                                   'max_gray': float(delta_zero.max().item() * 255)}
    (a.out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
