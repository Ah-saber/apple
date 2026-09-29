"""Probe a four-phase output converted from a trained 36-phase model.

The four kernels are the mean of the corresponding 3x3 output subphases.
This preserves the unrounded RAW-grid block mean before clamping. It changes
the native 3x image, so quality and temporal behavior must be checked.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def four_phase_conv(source):
    if source.out_channels != 36:
        raise ValueError(f'Expected 36 phases, got {source.out_channels}')
    result = nn.Conv2d(source.in_channels, 4, source.kernel_size,
                       stride=source.stride, padding=source.padding,
                       dilation=source.dilation, groups=source.groups,
                       bias=source.bias is not None,
                       device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        for py in range(2):
            for px in range(2):
                indices = [(3*py+dy)*6 + 3*px+dx
                           for dy in range(3) for dx in range(3)]
                target = 2*py+px
                result.weight[target].copy_(source.weight[indices].mean(0))
                if source.bias is not None:
                    result.bias[target].copy_(source.bias[indices].mean(0))
    return result


class FourPhase(nn.Module):
    def __init__(self, fast, gate_bias=0.5):
        super().__init__()
        self.fast = fast
        self.phase = four_phase_conv(fast.model.upsample[0])
        self.gate_bias = float(gate_bias)

    def forward(self, stack, context, context_box):
        fast, model = self.fast, self.fast.model
        current = stack[:, -1:]
        gate_input = torch.cat((stack, model._trajectory_features(stack)), dim=1)
        joined = fast.second(F.relu(fast.first(fast.half_input(gate_input)))).float()
        correction, logit = joined[:, :1], joined[:, 1:]
        denoised = current.float() + correction * (1. - torch.sigmoid(logit+self.gate_bias))
        features = model.body(model.head(model.down(fast.half_input(denoised))))
        reference = model.global_reference(fast.half_input(context), context_box,
                                           features.shape[-2:]).to(features.dtype)
        phases = self.phase(model.tail(features + reference)).float()
        sensor = F.pixel_shuffle(phases, 2)
        return F.interpolate(sensor, scale_factor=3, mode='nearest')


def measure(fn, warmup, repeat):
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return result, {'mean_ms': float(np.mean(times)),
                    'median_ms': float(np.median(times)),
                    'p95_ms': float(np.percentile(times, 95)),
                    'samples': len(times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    sys.path[:0] = [str(args.code / 'src'), str(args.runtime),
                    str(Path(__file__).resolve().parent)]
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config
    from fast_nine_inference import build_fast_model

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    fast = build_fast_model(config, state['model'], inference_model).cuda().eval()
    compact = FourPhase(fast).cuda().eval()
    data = dataset_for_config(config, 'val')
    row = data.records[0]
    stack = data.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
    context, _ = data.context_for(row)
    context = context[None].cuda()
    rt, rl, rh, rw = allowed_region(row)
    box = torch.tensor([[-rt/rh, -rl/rw, (1024-rt)/rh, (1280-rl)/rw]],
                       dtype=torch.float32, device='cuda')
    original, original_time = measure(lambda: fast(stack, context, box), 20, 80)
    reduced, reduced_time = measure(lambda: compact(stack, context, box), 20, 80)
    native_diff = (original.float()-reduced.float()).abs()*255
    original_sensor = F.avg_pool2d(original.float().clamp(0,1), 3, 3)
    reduced_sensor = F.avg_pool2d(reduced.float().clamp(0,1), 3, 3)
    raw_diff = (original_sensor-reduced_sensor).abs()*255
    report = {'scene': row['scene_id'], 'checkpoint': str(args.checkpoint),
              'baseline_fp16': original_time, 'four_phase_fp16': reduced_time,
              'native_mean_abs_gray': float(native_diff.mean()),
              'native_max_abs_gray': float(native_diff.max()),
              'raw_mean_abs_gray': float(raw_diff.mean()),
              'raw_max_abs_gray': float(raw_diff.max())}
    if args.compile:
        torch._inductor.config.triton.cudagraphs = False
        compiled = torch.compile(compact, fullgraph=True,
                                 options={'triton.cudagraphs': False})
        compiled_output, compiled_time = measure(
            lambda: compiled(stack, context, box), 20, 80)
        report['four_phase_compiled_fp16'] = compiled_time
        report['compiled_vs_source_mean_abs_gray'] = float(
            ((compiled_output-reduced).abs()*255).mean())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
