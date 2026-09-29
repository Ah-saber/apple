"""Screen low-resolution reference replacement on real complete frames."""
import argparse
import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lowres_reference import load_candidate, prepare_inputs
from output_candidates import load_candidate as load_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trained', action='store_true')
    parser.add_argument('--kernels', nargs='+', type=int, default=[1, 3, 5])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frame = 20 if args.scene == 'ordinary' else 60
        vector = np.load(args.v09_run / 'v08/test_vectors' /
                         f'{args.scene}_frame_{frame}.npz')
        raw = torch.from_numpy(vector['nine_raw']).cuda()
        context = torch.from_numpy(vector['reference_thumb']).cuda()
        baseline = load_output(args.scene, 'rows32', args.v09_run)
        raw, context = prepare_inputs(baseline, raw, context)
        rows = []
        with torch.inference_mode():
            standard = baseline(raw, context).float()
            for kernel in args.kernels:
                model = load_candidate(args.scene, kernel, 'rows32',
                                       args.v09_run, trained=args.trained)
                result = model(raw, context).float()
                delta = (result - standard).abs()
                row = {'scene': args.scene, 'kernel': kernel,
                       'trained': args.trained,
                       'mean_abs_gray': float(delta.mean()),
                       'p95_abs_gray': float(torch.quantile(delta.flatten(), .95)),
                       'max_abs_gray': float(delta.max()),
                       'output_mean_gray': float(result.mean()),
                       'baseline_mean_gray': float(standard.mean()),
                       'NPU_verified': False}
                rows.append(row)
                print(row, flush=True)
        (args.out / f'{args.scene}_lowref_probe_{"trained" if args.trained else "initial"}.json').write_text(
            json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
