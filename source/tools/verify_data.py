"""Independent CPU read-back and actual PyTorch Dataset/DataLoader verification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from ir_sr.data import RawDisplayDataset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    pairs = [json.loads(l) for l in (root / 'manifests/dataset_d0/pairs.jsonl').read_text().splitlines()]
    decoded = [json.loads(l) for l in (root / 'manifests/raw_v1/decoded.jsonl').read_text().splitlines()]
    sources = [json.loads(l) for l in (root / 'manifests/source_v1/sequences.jsonl').read_text().splitlines()]
    src = {(s['domain'], s['sequence_id']): s for s in sources}
    splits = {s: set((root / f'manifests/split_s0/{s}.txt').read_text().splitlines()) for s in ['train', 'val', 'test', 'diagnostic']}
    assert len(pairs) == 8669 and len(decoded) == 75
    assert len({p['sample_id'] for p in pairs}) == len(pairs)
    assert not (splits['train'] & splits['val'] or splits['train'] & splits['test'] or splits['val'] & splits['test'])
    assert splits['train'] | splits['val'] | splits['test'] == {p['sample_id'] for p in pairs}
    assert splits['diagnostic'] <= splits['train']
    group_splits, seq_splits = {}, {}
    for pair in pairs:
        split = next(s for s in ('train', 'val', 'test') if pair['sample_id'] in splits[s])
        for mapping, key in [(group_splits, pair['group_id']), (seq_splits, (pair['domain'], pair['sequence_id']))]:
            assert mapping.setdefault(key, split) == split
    raw_frames_compared = 0
    for r in decoded:
        s = src[r['domain'], r['sequence_id']]
        raw = np.load(root / r['raw_path'], mmap_mode='r', allow_pickle=False)
        assert raw.shape == (r['frame_count'], 1024, 1280) and raw.dtype == np.dtype('<u2')
        assert digest(root / r['raw_path']) == r['raw_sha256']
        with (root / s['path']).open('rb') as f:
            for t in (0, s['frame_count'] // 2, s['frame_count'] - 1):
                f.seek(t * s['frame_bytes'])
                a = np.frombuffer(f.read(s['frame_bytes']), dtype='<u2').reshape(s['stored_shape'])
                expected = a[:, 1280:] if s['domain'] == 'night' else a
                assert np.array_equal(raw[t], expected)
                raw_frames_compared += 1
    print(json.dumps({'event': 'raw_cache_readback', 'sequences': len(decoded), 'sampled_frames_exact': raw_frames_compared}), flush=True)
    # Every target read back; actual source pixels must remain identical.
    for i, pair in enumerate(pairs):
        t = pair['target']
        target = root / t['path']
        assert digest(target) == t['sha256']
        assert digest(root / t['source_path']) == t['source_sha256']
        with Image.open(target) as im:
            assert im.mode == 'L' and im.size == (1280, 1024)
            a = np.array(im)
        with Image.open(root / t['source_path']) as original:
            assert np.array_equal(a, np.asarray(original))
        if (i + 1) % 1000 == 0:
            print(json.dumps({'event': 'targets_verified', 'frames': i + 1}), flush=True)
    train = RawDisplayDataset(root, 'train')
    checks = []
    for domain in ('normal', 'adverse', 'night'):
        ids = [i for i, r in enumerate(train.records) if r['domain'] == domain]
        for i in [ids[0], ids[len(ids) // 2], ids[-1]]:
            item = train[i]
            assert item['raw'].shape == (1, 128, 128) and item['gt'].shape == (1, 384, 384)
            assert item['raw'].dtype == torch.float32 and item['gt'].dtype == torch.float32
            assert torch.isfinite(item['raw']).all() and torch.isfinite(item['gt']).all()
            assert item['gt'].min() >= 0 and item['gt'].max() <= 1
            r = train.records[i]
            raw = np.load(root / r['raw']['path'], mmap_mode='r')[r['frame_id']]
            top, left, height, width = item['crop_tlhw']
            # Independently evaluate several 3x3 DN blocks in float64.
            norm = train.recipe['raw_normalization']
            for yy, xx in ((0, 0), (37, 61), (127, 127)):
                expected = (raw[top + 3*yy:top + 3*yy + 3, left + 3*xx:left + 3*xx + 3].astype(np.float64).mean() - norm['offset']) / norm['scale']
                assert abs(float(item['raw'][0, yy, xx]) - expected) < 1e-5
            with Image.open(root / r['target']['path']) as im:
                gt = np.asarray(im)[top:top+height, left:left+width]
                assert np.array_equal(np.rint(item['gt'][0].numpy() * 255).astype(np.uint8), gt)
            full = train.full_raw(i)
            assert full.shape == (1, 1024, 1280) and torch.isfinite(full).all()
            checks.append({'sample_id': item['sample_id'], 'crop': item['crop_tlhw']})
    # Real multiprocessing DataLoader, no GPU tensors or pinned-memory transfers.
    loader = DataLoader(RawDisplayDataset(root, 'train'), batch_size=2, num_workers=2,
                        persistent_workers=False, pin_memory=False)
    batch = next(iter(loader))
    assert batch['raw'].shape == (2, 1, 128, 128) and batch['gt'].shape == (2, 1, 384, 384)
    validation = {}
    for split in ('val', 'test', 'diagnostic'):
        ds = RawDisplayDataset(root, split)
        item = ds[0]
        assert item['raw'].shape == (1, 340, 426) and item['gt'].shape == (1, 1020, 1278)
        validation[split] = {'samples': len(ds), 'raw_shape': list(item['raw'].shape), 'gt_shape': list(item['gt'].shape)}
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed', 'raw_caches_sha256_verified': 75,
              'raw_frames_exactly_compared': raw_frames_compared,
              'all_targets_sha256_and_pixels_verified': len(pairs),
              'train_sample_checks': checks, 'dataloader_workers': 2,
              'dataloader_raw_shape': list(batch['raw'].shape),
              'dataloader_gt_shape': list(batch['gt'].shape), 'validation': validation,
              'split_counts': {k: len(v) for k, v in splits.items()},
              'capture_groups_disjoint': True, 'sequences_disjoint': True,
              'independent_scene_holdout': False, 'cuda_initialized': torch.cuda.is_initialized(),
              'torch': torch.__version__, 'numpy': np.__version__}
    out = root / 'qa/dataset_d0/verification.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
