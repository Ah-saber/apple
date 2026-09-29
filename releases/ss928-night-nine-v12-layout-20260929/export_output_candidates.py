"""Export full and reduced-input output layouts using the frozen v0.11 weights."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from output_candidates import load_candidate, load_v11_expanded, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--layouts', nargs='+', required=True)
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
        raw = torch.from_numpy(vector['nine_raw']).cuda()
        context = torch.from_numpy(vector['reference_thumb']).cuda()
        baseline = load_v11_expanded(args.scene, args.v09_run).eval()
        raw, context = prepare_inputs(baseline, raw, context)
        with torch.inference_mode():
            reference = baseline(raw, context).float()
            rows = []
            for layout in args.layouts:
                model = load_candidate(args.scene, layout, args.v09_run).eval()
                value = model(raw, context).float()
                difference = (value - reference).abs()
                row = {'scene': args.scene, 'layout': layout,
                       'source_mean_abs_gray': float(difference.mean()),
                       'source_max_abs_gray': float(difference.max()),
                       'source_exact': bool(torch.equal(value, reference)),
                       'full_output_shape': list(value.shape),
                       'NPU_verified': False, 'files': []}
                assert value.shape == (1, 1, 3072, 3840), row
                for small in (False, True):
                    height, width = (128, 128) if small else (1024, 1280)
                    x = torch.full((1, 9, height, width), .3, device='cuda')
                    c = torch.full((1, 1, 64, 64), .3, device='cuda')
                    x, c = prepare_inputs(model, x, c)
                    suffix = '_small' if small else ''
                    path = args.out / f'{args.scene}_{layout}{suffix}.onnx'
                    torch.onnx.export(model, (x, c), str(path),
                        input_names=['nine_raw', 'reference_thumb'],
                        output_names=['display_gray'], opset_version=17,
                        dynamo=False)
                    graph, _ = materialize(onnx.load(str(path)))
                    onnx.checker.check_model(graph)
                    onnx.save(graph, str(path))
                    output_shape = [d.dim_value for d in
                        graph.graph.output[0].type.tensor_type.shape.dim]
                    assert output_shape == [1, 1, height * 3, width * 3]
                    row['files'].append({'name': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'shape': output_shape, 'nodes': len(graph.graph.node)})
                rows.append(row)
                print(row, flush=True)
                del model, value
                torch.cuda.empty_cache()
            (args.out / f'{args.scene}_manifest.json').write_text(
                json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
