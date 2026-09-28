"""Compare full-resolution four-channel output graphs with their 16-channel sources."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def run(path, vector, options):
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    feed = {meta.name: vector[meta.name].astype(
        np.float16 if meta.type == 'tensor(float16)' else np.float32)
        for meta in session.get_inputs()}
    return session.run(None, feed)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--aggressive-old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    rows = []
    for scene, front, reference in (
        ('ordinary', 'trim_t6', 'keep02_trained'),
        ('ordinary', 'trim_t6', 'keep1_projected'),
        ('special', 'trim_t6_o20', None),
    ):
        frame = 20 if scene == 'ordinary' else 60
        vector = np.load(args.vectors / f'{scene}_frame_{frame}.npz')
        prefix = f'{scene}_{front}' + (f'__{reference}' if reference else '')
        for mode in ('expand', 'tile') if reference != 'keep1_projected' else ('expand',):
            source_dir = args.aggressive_old if reference == 'keep1_projected' else args.old
            old = source_dir / f'{prefix}__{mode}_static.onnx'
            new = args.new / f'{prefix}__{mode}4_static.onnx'
            original = run(old, vector, options)
            trimmed = run(new, vector, options)
            delta = np.abs(original.astype(np.float32) - trimmed.astype(np.float32))
            graph = onnx.load(str(new))
            weights = [tensor for tensor in graph.graph.initializer
                       if tensor.name == 'output.conv.weight']
            assert len(weights) == 1 and weights[0].dims[0] == 4
            row = {'scene': scene, 'frame': frame, 'mode': mode,
                   'source': old.name, 'candidate': new.name,
                   'shape': list(trimmed.shape),
                   'exact_same_backend': bool(np.array_equal(original, trimmed)),
                   'different_pixels': int(np.count_nonzero(delta)),
                   'max_abs_gray': float(delta.max()),
                   'output_conv_channels': weights[0].dims[0]}
            assert row['exact_same_backend'], row
            rows.append(row)
            print(row, flush=True)
    (args.new / 'output4_full_verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
