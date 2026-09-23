"""Create a task-local, pixel-exact uint8 cache from frozen D1 training PNGs."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ir_sr.training import atomic_json, sha, validate_data_lock


def cache_one(payload):
    data_root, destination, key, records = payload
    root, destination = Path(data_root), Path(destination)
    records = sorted(records, key=lambda r: r['frame_id'])
    assert [r['frame_id'] for r in records] == list(range(len(records)))
    path = destination / key / 'target_u8.npy'
    path.parent.mkdir(parents=True, exist_ok=False)
    cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                     shape=(len(records), 1024, 1280))
    hashes = []
    for record in records:
        source = root / record['target']['path']
        assert sha(source) == record['target']['sha256']
        with Image.open(source) as im:
            assert im.mode == 'L' and im.size == (1280, 1024)
            pixels = np.asarray(im)
            cache[record['frame_id']] = pixels
            assert np.array_equal(cache[record['frame_id']], pixels)
        hashes.append(record['target']['sha256'])
    cache.flush()
    del cache
    return {'key': key, 'path': path.relative_to(destination).as_posix(),
            'shape': [len(records), 1024, 1280], 'dtype': 'uint8', 'sha256': sha(path),
            'source_target_hashes': hashes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    start = time.monotonic()
    code = Path(__file__).resolve().parents[1]
    lock = validate_data_lock(args.root, code)
    args.output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for line in (args.root / 'manifests/dataset_d1/pairs.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'train':
            groups[row['domain'] + '/' + row['sequence_id']].append(row)
    payloads = [(str(args.root), str(args.output), key, rows) for key, rows in sorted(groups.items())]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(cache_one, payloads):
            results.append(row)
            print(json.dumps({'event': 'target_sequence_cached', 'count': len(results), 'total': len(groups)}), flush=True)
    assert sum(r['shape'][0] for r in results) == 6509
    manifest = {'status': 'complete', 'dataset': 'dataset_d1', 'data_lock_sha256': lock,
                'frames': 6509, 'sequences': results, 'elapsed_seconds': time.monotonic() - start,
                'source_pixels_and_hashes_verified': True}
    atomic_json(args.output / 'cache_index.json', manifest)
    print(json.dumps({'status': 'complete', 'frames': 6509, 'sequences': len(results),
                      'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
