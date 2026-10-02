"""Read-only deployment hashes, pinned input identities and Python syntax."""
import ast
import json
from pathlib import Path
from prepare_sample import sha, load


def main():
    bundle = Path(__file__).resolve().parents[1]
    manifest = load(bundle/'MANIFEST.json')
    for row in manifest['files']:
        path = (bundle/row['path']).resolve()
        assert path.is_relative_to(bundle), row['path']
        assert path.stat().st_size == row['bytes'], row['path']
        assert sha(path) == row['sha256'], row['path']
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=row['path'])
    index = load(bundle/'MODEL_INDEX.json')
    assert len(index) == 14 and len({v['scene'] for v in index}) == 7
    for scene in {v['scene'] for v in index}:
        assert {v['role'] for v in index if v['scene'] == scene} == {'control', 'candidate'}
    for row in index:
        assert sha(bundle/row['path']) == row['sha256'], row['path']
        assert row['outputs'] == [{'name': 'display_gray', 'dtype': 'FLOAT16', 'shape': [1, 1, 3072, 3840]}]
        assert not row['SDK_compiled'] and not row['NPU_measured']
    calibration = load(bundle/'CALIBRATION_INDEX.json')
    total = 0
    for scene, data in calibration['scene_data'].items():
        assert len(data['checks']) == 16
        assert data['calibration'] == 4 and data['test'] == 12
        for row in data['checks']:
            ids = row['history_frame_ids']
            assert len(ids) == 9 and max(ids) <= row['frame'], (scene, row['member'])
            if row['split'] == 'calibration':
                assert len(set(ids)) == 9, (scene, row['member'])
            total += 1
    assert total == 112
    print(json.dumps({'files_checked': len(manifest['files']), 'indexed_models': len(index),
                      'pinned_real_samples': total, 'SDK_compiled': False, 'NPU_measured': False}))


if __name__ == '__main__':
    main()
