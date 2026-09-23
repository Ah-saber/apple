"""Create D1 metadata/recipe only, reusing immutable D0 RAW and target caches."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from prepare_data import dump_new, put, sha


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = args.root
    code = Path(__file__).resolve().parents[1]
    start = time.monotonic()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip():
        raise RuntimeError('Committed clean code is required')
    plan = json.loads((code / 'configs/data/split_s1_plan.json').read_text())
    catalog = json.loads((code / 'manifests/scene_catalog_v1.json').read_text())
    old_lock = json.loads((code / 'manifests/dataset_d0.lock.json').read_text())
    for entry in old_lock['manifests']:
        if sha(root / entry['path']) != entry['sha256']:
            raise RuntimeError('D0 manifest changed: ' + entry['path'])
    if sha(root / 'manifests/dataset_d0/DATASET_READY.json') != old_lock['release_record_sha256']:
        raise RuntimeError('D0 release changed')
    pairs = load_lines(root / 'manifests/dataset_d0/pairs.jsonl')
    by_sequence = defaultdict(list)
    for pair in pairs:
        by_sequence[pair['domain'], pair['sequence_id']].append(pair)
    seqs = {(r['domain'], r['sequence_id']): r for r in catalog['sequences']}
    assert set(seqs) == set(by_sequence)
    for key, rows in by_sequence.items():
        assert len(rows) == seqs[key]['frames']
        assert {r['group_id'] for r in rows} == {seqs[key]['capture_group']}
    assignments = {r['capture_group']: plan['independent_group_assignments'].get(r['capture_group'], 'train')
                   for r in seqs.values()}
    spatial_groups = {r['capture_group'] for r in seqs.values() if r['scene_id'] in plan['spatial_scenes']}
    assert all(assignments[g] == 'train' for g in spatial_groups)
    views = []

    def add_view(pair, split, spatial=False):
        r = deepcopy(pair)
        r['source_sample_id'] = pair['sample_id']
        r['sample_id'] = pair['sample_id'] + '@' + split
        r['split'] = split
        r['scene_id'] = seqs[pair['domain'], pair['sequence_id']]['scene_id']
        r['evaluation_scope'] = ('training' if split == 'train' else
                                 ('spatial_development' if spatial else 'capture_group_development'))
        if split == 'train':
            r['train_roi_tlhw'] = (plan['spatial_train_roi_tlhw'] if pair['group_id'] in spatial_groups
                                    else [0, 0, 1024, 1280])
        else:
            r['eval_crop_tlhw'] = (plan['spatial_eval_crops_tlhw'][split] if spatial
                                   else plan['independent_eval_crop_tlhw'])
        views.append(r)

    for pair in pairs:
        if assignments[pair['group_id']] == 'train':
            add_view(pair, 'train')
    for scene in catalog['scene_ids']:
        spatial = scene in plan['spatial_scenes']
        for split in ('val', 'test'):
            keys = sorted(k for k, r in seqs.items() if r['scene_id'] == scene and
                          (spatial or assignments[r['capture_group']] == split))
            if not spatial:
                keys = keys[:1]
            if not keys:
                raise RuntimeError('Scene has no evaluation sequence: ' + scene)
            budget = plan['frames_per_scene_per_eval_split']
            for i, key in enumerate(keys):
                n = budget // len(keys) + (i < budget % len(keys))
                rows = sorted(by_sequence[key], key=lambda r: r['frame_id'])
                indices = np.linspace(0, len(rows) - 1, n, dtype=int).tolist()
                assert len(set(indices)) == n
                for index in indices:
                    add_view(rows[index], split, spatial)
    used_sources = {r['source_sample_id'] for r in views}
    reserved = [{'source_sample_id': p['sample_id'], 'scene_id': seqs[p['domain'], p['sequence_id']]['scene_id'],
                 'capture_group': p['group_id'], 'assigned_holdout': assignments[p['group_id']],
                 'reason': 'unused frames from held-out capture group; excluded from training'}
                for p in pairs if p['sample_id'] not in used_sources]
    assert all(r['assigned_holdout'] != 'train' for r in reserved)
    # Fit only actual training pixels, including the regional restrictions.
    histogram = np.zeros(65536, dtype=np.int64)
    train_sequences = {}
    for r in views:
        if r['split'] == 'train':
            key = (r['domain'], r['sequence_id'])
            train_sequences.setdefault(key, r)
    for i, (key, row) in enumerate(sorted(train_sequences.items())):
        raw = np.load(root / row['raw']['path'], mmap_mode='r', allow_pickle=False)
        top, left, height, width = row['train_roi_tlhw']
        assert raw.dtype == np.uint16 and raw.shape == (seqs[key]['frames'], 1024, 1280)
        # Full-frame histograms were verified during D0; restricted regions are rescanned.
        if row['train_roi_tlhw'] == [0, 0, 1024, 1280]:
            hp = (root / row['raw']['path']).parent / 'dn_histogram.json'
            for value, count in json.loads(hp.read_text()):
                histogram[value] += count
        else:
            for frame in raw:
                histogram += np.bincount(frame[top:top+height, left:left+width].reshape(-1), minlength=65536)
        print(json.dumps({'event': 'fit_training_pixels', 'sequence': i + 1, 'total': len(train_sequences)}), flush=True)
    values = np.arange(65536, dtype=np.float64)
    pixels = int(histogram.sum())
    mean = float(np.dot(values, histogram) / pixels)
    std = float(np.sqrt(np.dot((values - mean) ** 2, histogram) / pixels))
    expected_pixels = sum(len(by_sequence[k]) * r['train_roi_tlhw'][2] * r['train_roi_tlhw'][3]
                          for k, r in train_sequences.items())
    assert pixels == expected_pixels
    recipe = json.loads((root / 'recipes/preprocess_v1.json').read_text())
    recipe.update(version='preprocess_v2', dataset='dataset_d1', split='split_s1',
                  domain_sampling='equal seven scenes, then equal sequence, then uniform frame',
                  evaluation_batch_size=1,
                  raw_normalization={'kind': 'fixed_affine', 'offset': mean, 'scale': max(std, 1.0),
                                     'estimated_from': 'D1 training original RAW pixels inside allowed training regions only',
                                     'fit_pixels': pixels, 'clipping': False})
    destination = root / 'manifests/dataset_d1'
    dump_new(destination / 'pairs.jsonl', views, lines=True)
    dump_new(destination / 'reserved.jsonl', reserved, lines=True)
    dump_new(destination / 'scene_catalog.json', catalog)
    dump_new(root / 'recipes/preprocess_v2.json', recipe)
    dump_new(root / 'manifests/split_s1/protocol.json', plan)
    group_rows = [{'group_id': g, 'assignment': s, 'restricted_training_roi':
                   plan['spatial_train_roi_tlhw'] if g in spatial_groups else None}
                  for g, s in sorted(assignments.items())]
    dump_new(root / 'manifests/split_s1/groups.jsonl', group_rows, lines=True)
    summary = {}
    for split in ('train', 'val', 'test'):
        rows = [r for r in views if r['split'] == split]
        put(root / f'manifests/split_s1/{split}.txt', ''.join(r['sample_id'] + '\n' for r in rows).encode())
        summary[split] = dict(Counter(r['scene_id'] for r in rows))
        for scene in catalog['scene_ids']:
            selected = [r['sample_id'] for r in rows if r['scene_id'] == scene]
            assert selected
            if split != 'train':
                assert len(selected) == plan['frames_per_scene_per_eval_split']
            put(root / f'manifests/split_s1/by_scene/{scene}/{split}.txt', ''.join(x + '\n' for x in selected).encode())
    all_metadata = sorted(list(destination.glob('*')) + list((root / 'manifests/split_s1').rglob('*'))
                          + [root / 'recipes/preprocess_v2.json'])
    hashes = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in all_metadata if p.is_file()]
    dump_new(root / 'qa/dataset_d1/manifest_sha256.json', hashes)
    result = {'run_id': args.run_id, 'builder_commit': commit, 'completed_at_server_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': round(time.monotonic() - start, 3), 'scene_counts': summary,
              'counts': dict(Counter(r['split'] for r in views)), 'training_sequences': len(train_sequences),
              'unique_active_source_frames': len(used_sources), 'reserved_source_frames': len(reserved),
              'spatial_capture_groups': sorted(spatial_groups), 'raw_normalization': recipe['raw_normalization'],
              'source_frames': len(pairs), 'new_raw_or_target_files': 0, 'status': 'pending_independent_verification'}
    dump_new(root / 'qa/dataset_d1/build_summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
