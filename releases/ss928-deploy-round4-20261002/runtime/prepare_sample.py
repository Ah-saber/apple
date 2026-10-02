"""Materialize one hash-pinned real sample for the selected full ONNX graph."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--role', choices=['control', 'candidate'], required=True)
    p.add_argument('--sample', required=True, help='train_00.npz .. train_03.npz or test_00.npz .. test_11.npz')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    bundle = Path(__file__).resolve().parents[1]
    rows = [r for r in load(bundle/'MODEL_INDEX.json') if r['scene'] == a.scene and r['role'] == a.role]
    if len(rows) != 1:
        raise ValueError('Unknown or ambiguous scene/model role')
    model = rows[0]
    modelpath = (bundle/model['path']).resolve()
    if not modelpath.is_relative_to(bundle) or sha(modelpath) != model['sha256']:
        raise ValueError('Model path/hash does not match the delivery index')
    checks = load(bundle/'CALIBRATION_INDEX.json')['scene_data'][a.scene]['checks']
    samples = [r for r in checks if Path(r['member']).name == a.sample]
    if len(samples) != 1:
        raise ValueError('Sample is not one of the pinned real calibration/test inputs')
    sample = samples[0]
    root = a.data_root.resolve()
    source = (root/sample['member']).resolve()
    if not source.is_relative_to(root) or sha(source) != sample['sha256']:
        raise ValueError('Real source sample hash mismatch; do not use old diagnostic inputs')
    ids = sample['history_frame_ids']
    if len(ids) != 9 or max(ids) > sample['frame']:
        raise ValueError('History is not causal')
    if sample['split'] == 'calibration' and len(set(ids)) != 9:
        raise ValueError('Calibration must contain nine distinct real history frames')
    arrays = []
    with np.load(source, allow_pickle=False) as data:
        if [r['name'] for r in model['inputs']] != ['nine_raw', 'reference_thumb']:
            raise ValueError('Unexpected model input interface')
        for declaration in model['inputs']:
            name = declaration['name']
            value = data[name]
            if list(value.shape) != declaration['shape'] or not np.isfinite(value).all():
                raise ValueError('Input shape or finite-value check failed: '+name)
            dtype = np.dtype({'FLOAT16': '<f2', 'FLOAT': '<f4'}[declaration['dtype']])
            value = np.ascontiguousarray(value, dtype=dtype)
            if not np.isfinite(value).all():
                raise ValueError('Precision conversion produced nonfinite input: '+name)
            arrays.append((name, value))
    a.out.mkdir(parents=True, exist_ok=False)
    files = []
    for name, value in arrays:
        path = a.out/(name+'.bin')
        value.tofile(path)
        files.append({'name': name, 'path': path.name, 'dtype': value.dtype.str,
                      'shape': list(value.shape), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    metadata = {'scene': a.scene, 'role': a.role, 'model_path': model['path'], 'model_sha256': model['sha256'],
                'source': str(source), 'source_member': sample['member'], 'source_sha256': sample['sha256'],
                'split': sample['split'], 'frame': sample['frame'], 'history_frame_ids': ids,
                'layout': 'NCHW', 'byte_order': 'little-endian', 'normalization_changed': False,
                'inputs': files, 'outputs': model['outputs']}
    (a.out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'scene': a.scene, 'role': a.role, 'sample': a.sample, 'inputs': files}))


if __name__ == '__main__':
    main()
