"""Test the proven night convolution fusion on current full-reference weights."""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def timing(fn, warmup=20, repeats=80):
    with torch.inference_mode():
        for _ in range(warmup):
            y = fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            y = fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
    return y, {'median_ms': float(np.median(samples)),
               'p95_ms': float(np.percentile(samples, 95))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compile', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.code / 'src'), str(a.runtime)]
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config
    from collapse_output_shuffles import collapse_output_shuffles
    from fused_nine import FusedNine
    from trajectory_batched_stats import trajectory_features_batched_stats

    class CurrentFusedNine(FusedNine):
        def forward(self, stack, context, context_box):
            m = self.model
            current = stack[:, -1:]
            gate_input = torch.cat((stack, m._trajectory_features(stack)), dim=1)
            joined = self.second(F.relu(self.first(self.half_input(gate_input)))).float()
            correction = joined[:, :1]
            gate = torch.sigmoid(joined[:, 1:])
            denoised = current.float() + correction * (1. - gate)
            features = m.body(m.head(m.down(self.half_input(denoised))))
            reference = m.global_reference(self.half_input(context), context_box,
                                           features.shape[-2:]).to(features.dtype)
            packed = m.upsample[0](m.tail(features + reference))
            return F.pixel_shuffle(packed.float(), 6).clamp(0, 1)

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
    context = context[None].cuda()
    rt, rl, rh, rw = allowed_region(row)
    box = torch.tensor([[-rt/rh, -rl/rw, (1024-rt)/rh, (1280-rl)/rw]],
                       dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        y_original = model(x, context=context, context_box=box).float().clamp(0, 1)
    model._trajectory_features = types.MethodType(trajectory_features_batched_stats, model)
    y_fast, fast_time = timing(lambda: model(x, context=context, context_box=box))
    model = model.cpu()
    model = collapse_output_shuffles(model)
    fused = CurrentFusedNine(model, raw_basis=True, channels_last=True, dense_second=True).cuda().eval()
    y_fused, fused_time = timing(lambda: fused(x, context, box))
    comparison = {}
    for name, reference in (('fast', y_fast.clamp(0, 1)), ('original', y_original)):
        delta = (y_fused - reference).abs() * 255
        comparison[name] = {'mean_gray': float(delta.mean().item()),
                            'max_gray': float(delta.max().item()),
                            'changed_uint8_fraction': float((torch.round(y_fused*255) !=
                                                             torch.round(reference*255)).float().mean().item())}
    result = {'checkpoint': str(a.checkpoint), 'scene': row['scene_id'],
              'fast_fp32': fast_time, 'fused_fp16': fused_time, 'comparison': comparison}
    if a.compile:
        compiled = torch.compile(fused, fullgraph=True,
                                 options={'triton.cudagraphs': False})
        y_compiled, result['compiled_fp16'] = timing(lambda: compiled(x, context, box))
        delta = (y_compiled - y_fused).abs() * 255
        result['compiled_vs_fused'] = {
            'mean_gray': float(delta.mean().item()),
            'max_gray': float(delta.max().item()),
            'changed_uint8_fraction': float((torch.round(y_compiled*255) !=
                                             torch.round(y_fused*255)).float().mean().item())}
    (a.out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
