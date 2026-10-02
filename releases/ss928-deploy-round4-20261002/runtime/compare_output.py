"""Compare full raw FP16 board output with its exact-graph CPU reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_sample import sha, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-metadata', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--board-output', type=Path, required=True, help='Full little-endian FP16 display_gray, no SDK header')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    metadata = load(a.input_metadata)
    reference_metadata = load(a.reference.with_suffix('.json'))
    if reference_metadata['model_sha256'] != metadata['model_sha256']:
        raise ValueError('PC and board model identities differ')
    if reference_metadata.get('sample_sha256') != metadata['source_sha256']:
        raise ValueError('PC and board sample identities differ')
    for row in metadata['inputs']:
        path = a.input_metadata.parent/row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Materialized board input file changed: '+row['name'])
    declaration = metadata['outputs'][0]
    if len(metadata['outputs']) != 1 or declaration['dtype'] != 'FLOAT16':
        raise ValueError('Expected one full FP16 output')
    shape = declaration['shape']
    expected_bytes = int(np.prod(shape))*2
    if a.board_output.stat().st_size != expected_bytes:
        raise ValueError('Board output is not the complete raw FP16 tensor')
    actual = np.fromfile(a.board_output, dtype='<f2').reshape(shape).astype(np.float32)
    with np.load(a.reference, allow_pickle=False) as data:
        expected = data[declaration['name']].astype(np.float32)
    if list(expected.shape) != shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite output or wrong reference shape')
    delta = actual-expected
    bytes_delta = np.abs(actual.clip(0, 255).astype(np.uint8).astype(np.float32)-expected.clip(0, 255).astype(np.uint8).astype(np.float32))
    result = {'scene': metadata['scene'], 'role': metadata['role'], 'frame': metadata['frame'],
              'model_sha256': metadata['model_sha256'], 'source_sha256': metadata['source_sha256'],
              'reference_sha256': sha(a.reference), 'board_output_sha256': sha(a.board_output),
              'shape': shape, 'mean_abs_gray': float(np.abs(delta).mean()),
              'max_abs_gray': float(np.abs(delta).max()), 'bias_gray': float(delta.mean()),
              'truncated_byte_mean_abs_gray': float(bytes_delta.mean()),
              'truncated_byte_max_abs_gray': float(bytes_delta.max()),
              'quality_acceptance_automated': False, 'GT_quality_or_NPU_speed_proven': False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
