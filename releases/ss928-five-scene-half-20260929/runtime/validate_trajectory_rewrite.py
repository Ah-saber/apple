"""Check full-frame numerical equivalence of the fast nine-frame trajectory path."""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-scene', type=int, default=12)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.code / 'src'), str(a.runtime)]
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config
    from trajectory_batched_stats import trajectory_features_batched_stats

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    model = inference_model(config, state['model']).cuda().eval()
    original = model._trajectory_features
    fast = types.MethodType(trajectory_features_batched_stats, model)
    result = {'checkpoint': str(a.checkpoint), 'checkpoint_step': state['progress']['step'],
              'per_scene': a.per_scene, 'scenes': {}}
    with torch.inference_mode():
        for split in ('val', 'test'):
            ds = dataset_for_config(config, split)
            grouped = {}
            for row in ds.records:
                grouped.setdefault(row['scene_id'], []).append(row)
            for scene, rows in grouped.items():
                records = rows[:a.per_scene]
                stats = {'frames': 0, 'fast_max_gray': 0., 'zero_max_gray': 0.,
                         'fast_mean_gray': [], 'zero_mean_gray': [],
                         'fast_changed_uint8_fraction': [], 'zero_changed_uint8_fraction': []}
                for row in records:
                    x = ds.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
                    context, _ = ds.context_for(row)
                    rt, rl, rh, rw = allowed_region(row)
                    box = torch.tensor([[-rt/rh, -rl/rw, (1024-rt)/rh, (1280-rl)/rw]],
                                       dtype=torch.float32, device='cuda')
                    context = context[None].cuda()
                    def forward(method):
                        model._trajectory_features = method
                        return model(x, context=context, context_box=box).float().clamp(0, 1)
                    y_original = forward(original)
                    y_fast = forward(fast)
                    zero = torch.zeros((x.shape[0], 4, x.shape[-2], x.shape[-1]),
                                       dtype=x.dtype, device=x.device)
                    y_zero = forward(lambda _x: zero)
                    rounded_original = torch.round(y_original * 255)
                    for label, value in (('fast', y_fast), ('zero', y_zero)):
                        delta = (value - y_original).abs() * 255
                        stats[label+'_max_gray'] = max(stats[label+'_max_gray'], float(delta.max().item()))
                        stats[label+'_mean_gray'].append(float(delta.mean().item()))
                        stats[label+'_changed_uint8_fraction'].append(
                            float((torch.round(value * 255) != rounded_original).float().mean().item()))
                    stats['frames'] += 1
                for label in ('fast', 'zero'):
                    stats[label+'_mean_gray'] = float(np.mean(stats[label+'_mean_gray']))
                    stats[label+'_changed_uint8_fraction'] = float(
                        np.mean(stats[label+'_changed_uint8_fraction']))
                result['scenes'][split+'__'+scene] = stats
                print(split, scene, json.dumps(stats), flush=True)
    (a.out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
