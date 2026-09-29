"""Time model-only full-frame GPU inference for current scene candidates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path('/data/zhangbenzhuang/huawei_sr')
RUNS = ROOT / 'runs'
PREFIX = 'SS928-FIVE-SCENE-NINE-SCRATCH-20260929-'
CHECKPOINTS = {
    'day_original': RUNS / 'SS928-QUALITY-PHASE-20260924-DAY-PHASE02-2K/checkpoints/step_000002000.pt',
    'day_nine_fullnight': RUNS / f'{PREFIX}DAY-FULLNIGHT-20K/checkpoints/best.pt',
    'light_medium_original': RUNS / 'SS928-QUALITY-V3-20260924-LIGHT_MEDIUM-ABSOLUTE-2K/checkpoints/step_000002000.pt',
    'light_medium_nine_basic': RUNS / f'{PREFIX}LIGHT-MEDIUM-20K/checkpoints/best.pt',
    'light_medium_nine_fullnight': RUNS / f'{PREFIX}LIGHT-FULLNIGHT-20K/checkpoints/best.pt',
    'heavy_c32_original': RUNS / 'SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt',
    'heavy_c32_nine_basic': RUNS / f'{PREFIX}HEAVY-C32-20K/checkpoints/best.pt',
    'heavy_c32_nine_fullnight': RUNS / f'{PREFIX}HEAVY-FULLNIGHT-20K/checkpoints/best.pt',
    'c32_nine_fullnight': RUNS / f'{PREFIX}C32-FULLNIGHT-20K/checkpoints/best.pt',
    'c32_nine_fullnight_low_lr': RUNS / f'{PREFIX}C32-FULLNIGHT-LR1E4-20K/checkpoints/best.pt',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tags', nargs='+', choices=tuple(CHECKPOINTS), default=tuple(CHECKPOINTS))
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=80)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code / 'src'))
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = torch.cuda.get_device_properties(0)
    output = {'protocol': 'Model forward only; real full 1024x1280 RAW input, batch 1, 3072x3840 output; FP32, TF32 off; CUDA events and synchronization; data loading, transfer and rendering excluded; concurrent server training active.',
              'gpu': gpu.name, 'torch': torch.__version__, 'results': {}}
    for tag in args.tags:
        checkpoint = CHECKPOINTS[tag]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = state['config']
        model = inference_model(config, state['model']).cuda().eval()
        ds = dataset_for_config(config, 'val')
        row = ds.records[0]
        if config.get('input_frames', 1) == 9:
            x = ds.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
        else:
            norm = ds.normalization_for(row)
            raw = (ds._raw(row).astype(np.float32) - norm['offset']) / norm['scale']
            x = torch.from_numpy(raw.copy())[None, None].cuda()
        context, _ = ds.context_for(row)
        rt, rl, rh, rw = allowed_region(row)
        box = torch.tensor([[-rt/rh, -rl/rw, (1024-rt)/rh, (1280-rl)/rw]],
                           dtype=torch.float32, device='cuda')
        context = context[None].cuda()
        with torch.inference_mode():
            for _ in range(args.warmup):
                y = model(x, context=context, context_box=box)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                y = model(x, context=context, context_box=box)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
        assert tuple(y.shape) == (1, 1, 3072, 3840), (tag, y.shape)
        item = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'scene': row['scene_id'], 'input_frames': x.shape[1],
                'input_shape': list(x.shape), 'output_shape': list(y.shape),
                'n': len(samples), 'median_ms': float(np.median(samples)),
                'p95_ms': float(np.percentile(samples, 95)),
                'mean_ms': float(np.mean(samples)), 'min_ms': float(np.min(samples)),
                'max_ms': float(np.max(samples))}
        output['results'][tag] = item
        print(tag, json.dumps({k: item[k] for k in ('n','median_ms','p95_ms','min_ms','max_ms')}), flush=True)
        del model, ds, x, y, state, context, box
        torch.cuda.empty_cache()
    (args.out / 'report.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
