"""Export 12-channel pre-projection reference resize controls."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reference_layout import load_candidate, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--output-layout', default='rows32')
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.v09_run / 'code_snapshot/tools'))
    sys.path.insert(0, '/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
    import onnx
    from materialize_aliases import materialize

    with (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frame = 20 if args.scene == 'ordinary' else 60
        vector = np.load(args.v09_run / 'v08/test_vectors' /
                         f'{args.scene}_frame_{frame}.npz')
        x = torch.from_numpy(vector['nine_raw']).cuda()
        context = torch.from_numpy(vector['reference_thumb']).cuda()
        original = load_candidate(args.scene, 'before16', args.output_layout, args.v09_run)
        candidate = load_candidate(args.scene, 'after12', args.output_layout, args.v09_run)
        x, context = prepare_inputs(original, x, context)
        with torch.inference_mode():
            base = original(x, context).float()
            value = candidate(x, context).float()
        difference = (base - value).abs()
        row = {'scene': args.scene, 'frame': frame,
               'output_layout': args.output_layout,
               'source_mean_abs_gray': float(difference.mean()),
               'source_max_abs_gray': float(difference.max()),
               'full_output_shape': list(value.shape),
               'NPU_verified': False, 'files': []}
        for small in (False, True):
            height, width = (128, 128) if small else (1024, 1280)
            ax = torch.full((1, 9, height, width), .3, device='cuda')
            ac = torch.full((1, 1, 64, 64), .3, device='cuda')
            ax, ac = prepare_inputs(candidate, ax, ac)
            path = args.out / (f'{args.scene}_reference_after12_{args.output_layout}' +
                ('_small.onnx' if small else '.onnx'))
            torch.onnx.export(candidate, (ax, ac), str(path),
                input_names=['nine_raw', 'reference_thumb'],
                output_names=['display_gray'], opset_version=17,
                dynamo=False)
            graph, _ = materialize(onnx.load(str(path)))
            onnx.checker.check_model(graph)
            onnx.save(graph, str(path))
            row['files'].append({'name': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'nodes': len(graph.graph.node)})
        (args.out / f'{args.scene}_reference_{args.output_layout}_manifest.json').write_text(
            json.dumps(row, indent=2))
        print(row, flush=True)


if __name__ == '__main__':
    main()
