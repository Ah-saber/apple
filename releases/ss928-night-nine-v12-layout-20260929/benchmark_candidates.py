"""Same-round complete-model GPU timing; never infer SS928 speed from it."""
import argparse
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from lowres_reference import load_candidate as load_lowref
from output_candidates import load_candidate as load_output, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frame = 20 if args.scene == 'ordinary' else 60
        vector = np.load(args.v09_run / 'v08/test_vectors' /
                         f'{args.scene}_frame_{frame}.npz')
        raw = torch.from_numpy(vector['nine_raw']).cuda()
        context = torch.from_numpy(vector['reference_thumb']).cuda()
        report = {'scene': args.scene, 'GPU': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'input_shape': list(raw.shape),
                  'output_shape': [1, 1, 3072, 3840],
                  'TF32': False, 'CUDA_graphs': False,
                  'NPU_verified': False, 'results': []}
        with torch.inference_mode():
            baseline = load_output(args.scene, 'rows32', args.v09_run)
            raw, context = prepare_inputs(baseline, raw, context)
            reference = baseline(raw, context).float()
            for case in args.cases:
                model = (load_lowref(args.scene, 5,
                    case.removeprefix('lowref_k5_'), args.v09_run, trained=True)
                    if case.startswith('lowref_k5_')
                    else load_output(args.scene, case, args.v09_run))
                source = model(raw, context)
                difference = (source.float() - reference).abs()
                torch.compiler.reset()
                torch._inductor.config.triton.cudagraphs = False
                compiled = torch.compile(model, fullgraph=True)
                candidate = compiled(raw, context)
                own_diff = (candidate.float() - source.float()).abs()
                for _ in range(30):
                    compiled(raw, context)
                samples = []
                for _ in range(3):
                    times = []
                    for _ in range(100):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        compiled(raw, context)
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                    samples.append(times)
                row = {'case': case, 'mean_ms': float(np.mean(samples)),
                       'p95_ms': float(np.percentile(samples, 95)),
                       'source_mean_gray_vs_safe': float(difference.mean()),
                       'source_max_gray_vs_safe': float(difference.max()),
                       'compiled_own_source_mean_gray': float(own_diff.mean()),
                       'compiled_own_source_max_gray': float(own_diff.max()),
                       'samples_ms': samples}
                report['results'].append(row)
                print('BENCH', args.scene, {key: val for key, val in row.items()
                                               if key != 'samples_ms'}, flush=True)
                (args.out / f'{args.scene}_benchmark.json').write_text(
                    json.dumps(report, indent=2))
                del compiled, model, source, candidate
                gc.collect()
                torch.compiler.reset()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
