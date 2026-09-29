"""Verify alternative full-output layouts in one ONNX Runtime backend."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

LAYOUTS = ('rows8', 'rows16', 'rows32', 'rows64', 'rows128',
           'direct6', 'native_deconv', 'native_nearest', 'native_separable')


def extract(source, target):
    inferred = target.with_name(target.stem + '_inferred.onnx')
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(source))),
              str(inferred))
    onnx.utils.extract_model(str(inferred), str(target),
                             ['/output/Cast_output_0'], ['display_gray'])
    inferred.unlink()
    graph = onnx.load(str(target))
    batch = graph.graph.input[0].type.tensor_type.shape.dim[0]
    batch.ClearField('dim_param')
    batch.dim_value = 1
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))
    return graph


def run(path, tensor, options):
    session = ort.InferenceSession(str(path), options,
                                   providers=['CPUExecutionProvider'])
    assert len(session.get_inputs()) == 1
    return session.run(None, {session.get_inputs()[0].name: tensor})[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    rng = np.random.default_rng(928)
    small = rng.uniform(-.3, .3, [1, 16, 64, 64]).astype(np.float16)
    full = rng.uniform(-.3, .3, [1, 16, 512, 640]).astype(np.float16)
    rows = []
    for scene in ('ordinary', 'special'):
        for scale, tensor in (('small', small), ('full', full)):
            reference = None
            for layout in LAYOUTS:
                # The reduced input source preserves 64x64 output features.
                source = args.source_dir / (f'{scene}_{layout}' +
                    ('_small.onnx' if scale == 'small' else '.onnx'))
                target = args.out / f'{scene}_{layout}_{scale}_output_only.onnx'
                graph = extract(source, target)
                meta = graph.graph.input[0]
                dims = [d.dim_value for d in meta.type.tensor_type.shape.dim]
                assert dims[1:] == list(tensor.shape[1:]), (target, dims)
                value = run(target, tensor, options)
                if reference is None:
                    assert layout == 'rows8'
                    reference = value
                delta = np.abs(value.astype(np.float32) -
                               reference.astype(np.float32))
                row = {'scene': scene, 'layout': layout, 'scale': scale,
                       'input_shape': list(tensor.shape),
                       'output_shape': list(value.shape),
                       'exact_same_backend': bool(np.array_equal(value, reference)),
                       'different_pixels': int(np.count_nonzero(delta)),
                       'max_abs_gray': float(delta.max()),
                       'output_nodes': len(graph.graph.node),
                       'output_op_types': sorted(set(n.op_type for n in
                                                     graph.graph.node)),
                       'NPU_verified': False}
                assert row['exact_same_backend'], row
                rows.append(row)
                print(row, flush=True)
    (args.out / 'verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
