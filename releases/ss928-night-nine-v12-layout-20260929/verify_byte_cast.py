"""Verify real-frame direct-cast ONNX semantics and quantify rounding shift."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--byte-dir', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    rows = []
    for scene, frame in (('ordinary', 20), ('special', 60)):
        vector = np.load(args.vectors / f'{scene}_frame_{frame}.npz')
        for layout in ('rows32', 'direct6', 'native_deconv'):
            source = ort.InferenceSession(str(args.source_dir /
                f'{scene}_{layout}.onnx'), options,
                providers=['CPUExecutionProvider'])
            candidate = ort.InferenceSession(str(args.byte_dir /
                f'{scene}_{layout}_cast_u8.onnx'), options,
                providers=['CPUExecutionProvider'])
            feed = {meta.name: vector[meta.name].astype(
                np.float16 if meta.type == 'tensor(float16)' else np.float32)
                for meta in source.get_inputs()}
            gray = source.run(None, feed)[0]
            actual = candidate.run(None, feed)[0]
            truncated = np.trunc(gray.astype(np.float32)).astype(np.uint8)
            even = np.rint(gray.astype(np.float32)).astype(np.uint8)
            delta = actual.astype(np.int16) - even.astype(np.int16)
            trunc_delta = actual.astype(np.int16) - truncated.astype(np.int16)
            row = {'scene': scene, 'frame': frame, 'layout': layout,
                   'shape': list(actual.shape),
                   'exact_truncation_same_backend': bool(np.array_equal(
                       actual, truncated)),
                   'different_from_truncation_fraction': float(np.mean(trunc_delta != 0)),
                   'max_abs_vs_truncation_gray': int(np.abs(trunc_delta).max()),
                   'changed_from_nearest_even_fraction': float(np.mean(delta != 0)),
                   'mean_bias_vs_nearest_even_gray': float(delta.mean()),
                   'max_abs_vs_nearest_even_gray': int(np.abs(delta).max()),
                   'NPU_verified': False}
            assert row['max_abs_vs_truncation_gray'] <= 1, row
            rows.append(row)
            print(row, flush=True)
    (args.byte_dir / 'real_frame_verification.json').write_text(
        json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
