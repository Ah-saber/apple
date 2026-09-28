"""Extract output-stage diagnostic graphs with the same full display dimensions."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


SOURCE = {
    'ordinary_rows32': 'ordinary_rows32.onnx',
    'ordinary_expand4': 'ordinary_trim_t6__keep02_trained__expand4_static.onnx',
    'ordinary_tile4': 'ordinary_trim_t6__keep02_trained__tile4_static.onnx',
    'special_rows32': 'special_rows32.onnx',
    'special_expand4': 'special_trim_t6_o20__expand4_static.onnx',
    'special_tile4': 'special_trim_t6_o20__tile4_static.onnx',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    rows = []
    reference = {}
    rng = np.random.default_rng(928)
    feature = rng.uniform(-.3, .3, [1, 16, 512, 640]).astype(np.float16)
    for key, name in SOURCE.items():
        source = args.source_dir / name
        inferred = args.output_dir / f'{key}_shape_inferred.onnx'
        target = args.output_dir / f'{key}_output_only.onnx'
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(source))), str(inferred))
        onnx.utils.extract_model(str(inferred), str(target),
                                 ['/output/Cast_output_0'], ['display_gray'])
        inferred.unlink()
        graph = onnx.load(str(target))
        onnx.checker.check_model(graph)
        meta = graph.graph.input[0]
        dims = [dim.dim_value for dim in meta.type.tensor_type.shape.dim]
        assert dims[1:] == [16, 512, 640] and meta.type.tensor_type.elem_type == 10
        session = ort.InferenceSession(str(target), options,
                                       providers=['CPUExecutionProvider'])
        output = session.run(None, {meta.name: feature})[0]
        assert output.shape == (1, 1, 3072, 3840)
        scene = key.split('_', 1)[0]
        if key.endswith('rows32'):
            reference[scene] = output
        exact = bool(np.array_equal(output, reference[scene]))
        assert exact, key
        row = {'key': key, 'source': name, 'micrograph': target.name,
               'input_shape': [1, 16, 512, 640], 'input_dtype': 'float16',
               'output_shape': list(output.shape), 'output_dtype': str(output.dtype),
               'nodes': len(graph.graph.node), 'valid_onnx': True,
               'exact_vs_rows32_output_stage': exact,
               'NPU_verified': False}
        rows.append(row)
        print(row, flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
