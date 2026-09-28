"""Full nine-frame, full 3x-output source comparison and GPU screening."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from direct_output_candidates import load_candidate, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    frame = 20 if args.scene == 'ordinary' else 60
    sample = np.load(args.v09_run / 'v08' / 'test_vectors' / f'{args.scene}_frame_{frame}.npz')
    raw = torch.from_numpy(sample['nine_raw']).cuda()
    context = torch.from_numpy(sample['reference_thumb']).cuda()
    report = {'scene': args.scene, 'frame': frame, 'gpu': torch.cuda.get_device_name(0),
              'input_shape': list(raw.shape), 'output_shape': [1, 1, 3072, 3840],
              'board_verified': False, 'results': []}
    names = ['rows32', 'direct_g1', 'direct_g2', 'direct_g4', 'direct_g8', 'direct_g16']
    with torch.inference_mode():
        baseline = None
        for name in names:
            model = load_candidate(args.scene, name, args.v09_run).eval()
            x, c = prepare_inputs(model, raw, context)
            output = model(x, c)
            if baseline is None:
                baseline = output.float()
            delta = (output.float() - baseline).abs()
            assert list(output.shape) == report['output_shape']
            for _ in range(20):
                model(x, c)
            timings = []
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x, c)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            entry = {'name': name, 'dtype': str(output.dtype), 'mean_abs_gray': float(delta.mean()),
                     'max_abs_gray': float(delta.max()), 'exact': bool(torch.equal(output.float(), baseline)),
                     'gpu_eager_mean_ms': float(np.mean(timings)),
                     'gpu_eager_p95_ms': float(np.percentile(timings, 95))}
            report['results'].append(entry)
            (args.output / f'{args.scene}_complete.json').write_text(json.dumps(report, indent=2))
            print(entry, flush=True)
            del model, output
            torch.cuda.empty_cache()
    print('COMPLETE', args.scene, flush=True)


if __name__ == '__main__':
    main()
