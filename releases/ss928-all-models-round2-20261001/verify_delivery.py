"""Verify indexed model hashes and preserved experiment files without Torch."""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    release = Path(__file__).resolve().parent
    repo = release.parent.parent
    report = repo / 'reports/ss928_all_models_round2_20260930'
    index = json.loads((release / 'MODEL_INDEX.json').read_text())
    assert len(index['models']) == 21
    assert len({v['scene'] for v in index['models']}) == 7
    for row in index['models']:
        path = (repo / row['path']).resolve()
        assert path.is_relative_to(repo)
        assert sha(path) == row['sha256'], row['path']
        assert row['outputs'][0]['shape'] == [1, 1, 3072, 3840]
        assert row['outputs'][0]['dtype'] == 'FLOAT16'
    frozen = json.loads((report / 'results/review_file_verification.json').read_text())
    for row in frozen['files']:
        path = (report / 'results' / row['path']).resolve()
        assert path.is_relative_to(report)
        assert path.stat().st_size == row['bytes'], row['path']
        assert sha(path) == row['sha256'], row['path']
    print(json.dumps({'indexed_graph_hashes': 21, 'frozen_file_hashes': len(frozen['files']),
                      'all_match': True, 'NPU_measured': False}))


if __name__ == '__main__':
    main()
