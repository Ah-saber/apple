"""Independently verify D1 scene coverage, spatial guards and actual CPU loading."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from ir_sr.data import RawDisplayDataset
from prepare_data import sha, dump_new


def rectangle_gap(a, b):
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    return max(bt - (at + ah), at - (bt + bh), bl - (al + aw), al - (bl + bw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    start = time.monotonic()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    base = Path(__file__).resolve().parents[1]
    lock = json.loads((base / 'manifests/dataset_d0.lock.json').read_text())
    for row in lock['manifests']:
        assert sha(root / row['path']) == row['sha256']
    assert sha(root / 'manifests/dataset_d0/DATASET_READY.json') == lock['release_record_sha256']
    hashes = json.loads((root / 'qa/dataset_d1/manifest_sha256.json').read_text())
    for row in hashes:
        assert sha(root / row['path']) == row['sha256']
    old_pairs = {r['sample_id']: r for r in map(json.loads, (root / 'manifests/dataset_d0/pairs.jsonl').read_text().splitlines())}
    records = list(map(json.loads, (root / 'manifests/dataset_d1/pairs.jsonl').read_text().splitlines()))
    assert len({r['sample_id'] for r in records}) == len(records)
    split_rows = {s: [r for r in records if r['split'] == s] for s in ('train', 'val', 'test')}
    counts = {s: Counter(r['scene_id'] for r in rows) for s, rows in split_rows.items()}
    assert len(counts['train']) == len(counts['val']) == len(counts['test']) == 7
    assert set(counts['train']) == set(counts['val']) == set(counts['test'])
    assert all(n == 12 for s in ('val', 'test') for n in counts[s].values())
    for r in records:
        original = old_pairs[r['source_sample_id']]
        assert r['raw'] == original['raw'] and r['target'] == original['target']
        assert (r['domain'], r['sequence_id'], r['frame_id'], r['group_id']) == (
            original['domain'], original['sequence_id'], original['frame_id'], original['group_id'])
    for split, rows in split_rows.items():
        ids = (root / f'manifests/split_s1/{split}.txt').read_text().splitlines()
        assert ids == [r['sample_id'] for r in rows]
        for scene in counts[split]:
            scene_ids = (root / f'manifests/split_s1/by_scene/{scene}/{split}.txt').read_text().splitlines()
            assert scene_ids == [r['sample_id'] for r in rows if r['scene_id'] == scene]
    # Cross-capture evaluation cannot share groups with any training record.
    train_groups = {r['group_id'] for r in split_rows['train']}
    heldout_groups = {}
    spatial_by_group = defaultdict(dict)
    for split in ('val', 'test'):
        for r in split_rows[split]:
            if r['evaluation_scope'] == 'capture_group_development':
                assert r['group_id'] not in train_groups
                assert heldout_groups.setdefault(r['group_id'], split) == split
            else:
                assert r['evaluation_scope'] == 'spatial_development'
                assert r['scene_id'] in ('weather_heavy_c32', 'night_special')
                spatial_by_group[r['group_id']][split] = r['eval_crop_tlhw']
    assert len(spatial_by_group) == 3
    guarded_rows = 0
    for r in split_rows['train']:
        if r['group_id'] in spatial_by_group:
            assert set(spatial_by_group[r['group_id']]) == {'val', 'test'}
            for crop in spatial_by_group[r['group_id']].values():
                assert rectangle_gap(r['train_roi_tlhw'], crop) >= 64
            guarded_rows += 1
    for crops in spatial_by_group.values():
        assert rectangle_gap(crops['val'], crops['test']) >= 64
    reserved = list(map(json.loads, (root / 'manifests/dataset_d1/reserved.jsonl').read_text().splitlines()))
    reserved_ids = {r['source_sample_id'] for r in reserved}
    active_ids = {r['source_sample_id'] for r in records}
    assert len(reserved_ids) == len(reserved) and not reserved_ids & active_ids
    assert reserved_ids | active_ids == set(old_pairs)
    assert all(r['capture_group'] not in train_groups for r in reserved)
    # Refit independently from all actual training pixels, without using cached histograms.
    seq_rows = {}
    for r in split_rows['train']:
        key = (r['domain'], r['sequence_id'])
        seq_rows.setdefault(key, []).append(r)
    pixel_count, centered_sum, centered_square_sum = 0, 0.0, 0.0
    for i, rows in enumerate(seq_rows.values()):
        row = rows[0]
        raw = np.load(root / row['raw']['path'], mmap_mode='r', allow_pickle=False)
        assert {r['frame_id'] for r in rows} == set(range(len(raw)))
        assert all(r['train_roi_tlhw'] == row['train_roi_tlhw'] for r in rows)
        top, left, height, width = row['train_roi_tlhw']
        for frame in raw:
            v = frame[top:top+height, left:left+width].astype(np.float64) - 5000.0
            centered_sum += float(v.sum())
            centered_square_sum += float(np.square(v).sum())
            pixel_count += v.size
        if (i + 1) % 10 == 0:
            print(json.dumps({'event': 'independent_normalization', 'sequences': i + 1}), flush=True)
    expected_mean = 5000.0 + centered_sum / pixel_count
    expected_std = float(np.sqrt(centered_square_sum / pixel_count - (centered_sum / pixel_count) ** 2))
    train = RawDisplayDataset(root, dataset_version='dataset_d1')
    norm = train.recipe['raw_normalization']
    assert pixel_count == norm['fit_pixels']
    assert abs(expected_mean - norm['offset']) < 1e-8 and abs(expected_std - norm['scale']) < 1e-8
    # All evaluation items, plus three actual training crops per profile.
    checked, shapes = [], defaultdict(set)

    def check_item(dataset, index):
        r = dataset.records[index]
        item = dataset[index]
        raw = np.load(root / r['raw']['path'], mmap_mode='r', allow_pickle=False)[r['frame_id']]
        top, left, height, width = item['crop_tlhw']
        if dataset.mode == 'train':
            rt, rl, rh, rw = r['train_roi_tlhw']
            assert rt <= top and rl <= left and top + height <= rt + rh and left + width <= rl + rw
        else:
            assert list(item['crop_tlhw']) == r['eval_crop_tlhw']
        expected = raw[top:top+height, left:left+width].astype(np.float64)
        expected = expected.reshape(height // 3, 3, width // 3, 3).mean(axis=(1, 3))
        expected = (expected - norm['offset']) / norm['scale']
        np.testing.assert_allclose(item['raw'][0].numpy(), expected, rtol=0, atol=1e-5)
        assert item['raw'].dtype == item['gt'].dtype == torch.float32
        assert torch.isfinite(item['raw']).all() and torch.isfinite(item['gt']).all()
        target_path = root / r['target']['path']
        assert sha(target_path) == r['target']['sha256']
        with Image.open(target_path) as im:
            gt = np.asarray(im)[top:top+height, left:left+width]
            np.testing.assert_array_equal(np.rint(item['gt'][0].numpy() * 255).astype(np.uint8), gt)
        assert item['scene_id'] == r['scene_id'] and item['evaluation_scope'] == r['evaluation_scope']
        shapes[r['split']].add(tuple(item['raw'].shape))
        checked.append(r['sample_id'])

    for scene in sorted(counts['train']):
        indices = [i for i, r in enumerate(train.records) if r['scene_id'] == scene]
        for i in (indices[0], indices[len(indices) // 2], indices[-1]):
            check_item(train, i)
    for split in ('val', 'test'):
        ds = RawDisplayDataset(root, split, dataset_version='dataset_d1')
        for i in range(len(ds)):
            check_item(ds, i)
        for scene in counts[split]:
            assert len(RawDisplayDataset(root, split, dataset_version='dataset_d1', scene_id=scene)) == 12
    # Dataset itself must reject bypasses of spatial inference context restrictions.
    ds = RawDisplayDataset(root, 'val', dataset_version='dataset_d1', scene_id='night_special')
    try:
        ds.full_raw(0)
    except ValueError:
        pass
    else:
        raise AssertionError('Spatial full-frame bypass accepted')
    try:
        RawDisplayDataset(root, 'val', mode='train', dataset_version='dataset_d1')
    except ValueError:
        pass
    else:
        raise AssertionError('Split mode bypass accepted')
    loader = DataLoader(train, batch_size=2, num_workers=2, persistent_workers=False, pin_memory=False)
    batch = next(iter(loader))
    assert batch['raw'].shape == (2, 1, 128, 128) and batch['gt'].shape == (2, 1, 384, 384)
    sampler = train.balanced_sampler()
    mass = {scene: sum(float(sampler.weights[i]) for i, r in enumerate(train.records) if r['scene_id'] == scene)
            for scene in counts['train']}
    assert all(abs(x - 1.0) < 1e-10 for x in mass.values())
    # D0 can still be loaded with its historical defaults.
    old = RawDisplayDataset(root, 'val')
    assert len(old) == 780 and old[0]['raw'].shape == (1, 340, 426)
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed', 'counts': {s: len(rows) for s, rows in split_rows.items()},
              'scene_counts': {s: dict(c) for s, c in counts.items()},
              'seven_scenes_in_all_splits': True, 'all_168_eval_items_checked': True,
              'train_items_checked': 21, 'restricted_training_frames': guarded_rows,
              'minimum_spatial_guard_hr': 64, 'all_related_capture_training_regions_checked': True,
              'independent_training_pixel_refit': {'pixels': pixel_count, 'offset': expected_mean, 'scale': expected_std},
              'unique_active_source_frames': len(active_ids), 'reserved_source_frames': len(reserved_ids),
              'input_shapes': {s: sorted(shapes[s]) for s in shapes},
              'dataloader_workers': 2, 'sampler_probability_mass_equal_across_seven_scenes': True,
              'd0_metadata_unchanged_and_loader_compatible': True,
              'spatial_evaluation_is_development_only': True, 'independent_scene_generalization': False,
              'cuda_initialized': torch.cuda.is_initialized(), 'elapsed_seconds': round(time.monotonic() - start, 3)}
    dump_new(root / 'qa/dataset_d1/verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
