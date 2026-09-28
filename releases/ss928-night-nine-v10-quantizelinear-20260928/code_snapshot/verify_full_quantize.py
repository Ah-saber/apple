"""Validate full-resolution QuantizeLinear output on frozen real test frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vectors', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    args = p.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    rows = []
    for scene, frame in (('ordinary', 20), ('special', 60)):
        graph = args.graphs / f'{scene}_rows32_quantizelinear_u8.onnx'
        session = ort.InferenceSession(str(graph), opts, providers=['CPUExecutionProvider'])
        source_session = ort.InferenceSession(str(args.source_dir / f'{scene}_rows32.onnx'),
            opts, providers=['CPUExecutionProvider'])
        sample = np.load(args.vectors / f'{scene}_frame_{frame}.npz')
        feed = {'nine_raw': sample['nine_raw'], 'reference_thumb': sample['reference_thumb']}
        for input_meta in session.get_inputs():
            if input_meta.type == 'tensor(float16)':
                feed[input_meta.name] = feed[input_meta.name].astype(np.float16)
            elif input_meta.type == 'tensor(float)':
                feed[input_meta.name] = feed[input_meta.name].astype(np.float32)
        actual = session.run(None, feed)[0]
        source_ort = source_session.run(None, feed)[0]
        expected_ort = np.rint(np.clip(source_ort.astype(np.float32), 0, 255)).astype(np.uint8)
        source = np.load(args.reference / f'{scene}_rows32.npz')['output']
        expected = np.rint(np.clip(source.astype(np.float32), 0, 255)).astype(np.uint8)
        row = {'scene': scene, 'frame': frame, 'output_shape': list(actual.shape),
               'exact_same_backend_uint8_equality': bool(np.array_equal(actual, expected_ort)),
               'different_pixels_same_backend': int(np.count_nonzero(actual != expected_ort)),
               'different_pixels_vs_pytorch_source': int(np.count_nonzero(actual != expected)),
               'max_byte_difference_vs_pytorch_source': int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max())}
        assert row['exact_same_backend_uint8_equality'], row
        rows.append(row)
        print(row, flush=True)
    (args.graphs.parent / 'quantize_full_verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
