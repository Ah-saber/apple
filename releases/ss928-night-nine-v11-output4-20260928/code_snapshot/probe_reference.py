"""Full-frame reference ablation, exactness and GPU screening."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from reference_candidates import load_candidate, prepare_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    p.add_argument('--v09-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        frame = 20 if a.scene == 'ordinary' else 60
        vec = np.load(a.v09_run / 'v08' / 'test_vectors' / f'{a.scene}_frame_{frame}.npz')
        x = torch.from_numpy(vec['nine_raw']).cuda()
        context = torch.from_numpy(vec['reference_thumb']).cuda()
        names = ['rows32', 'bilinear_wrapper', 'nearest_reference']
        if a.scene == 'ordinary':
            names.extend(['keep01', 'keep02', 'keep12', 'keep0', 'keep1', 'keep2', 'keep'])
        if a.cases:
            names = a.cases
        rows = []
        with torch.inference_mode():
            baseline = None
            for name in names:
                model = load_candidate(a.scene, name, a.v09_run).eval()
                xx, cc = prepare_inputs(model, x, context)
                result = model(xx, cc).float()
                if baseline is None:
                    baseline = result
                delta = (result - baseline).abs()
                for _ in range(10):
                    model(xx, cc)
                timings = []
                for _ in range(100):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    model(xx, cc)
                    end.record()
                    end.synchronize()
                    timings.append(begin.elapsed_time(end))
                row = {'case': name, 'shape': list(result.shape),
                       'mean_abs_gray_vs_rows32': float(delta.mean()),
                       'max_abs_gray_vs_rows32': float(delta.max()),
                       'same_pixel_fraction': float((delta == 0).float().mean()),
                       'gpu_eager_mean_ms': float(np.mean(timings)),
                       'gpu_eager_p95_ms': float(np.percentile(timings, 95))}
                rows.append(row)
                (a.output / f'{a.scene}_reference_probe.json').write_text(json.dumps(rows, indent=2))
                print(row, flush=True)
                del model, result
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
