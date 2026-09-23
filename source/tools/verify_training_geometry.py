"""Audit actual RAW/GT crop correspondence and unchanged held-out pixels on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
import torch
from ir_sr.data import RawDisplayDataset
from ir_sr.training import atomic_json, dataset_for_config, record_selected_data, validate_data_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    lock = validate_data_lock(config['data_root'], code)
    ds = dataset_for_config(config, 'train', use_cache=True)
    old_config = {k: v for k, v in config.items() if k != 'train_crop_hr'}
    legacy = dataset_for_config(old_config, 'train', use_cache=True)
    assert ds.records == legacy.records and ds.crop_indices == legacy.crop_indices
    height, width = config['train_crop_hr']
    expected_lr = [height // 3, width // 3]
    # Check ALL selected ROI bounds, then independently rebuild representative tensors.
    for r in ds.records:
        rt, rl, rh, rw = r.get('train_roi_tlhw', [0, 0, 1024, 1280])
        assert min(rt, rl) >= 0 and height <= rh and width <= rw
        assert rt + rh <= 1024 and rl + rw <= 1280
    selected = []
    for scene in config['scene_ids']:
        indices = [i for i, r in enumerate(ds.records) if r['scene_id'] == scene]
        selected += [indices[int(i)] for i in np.linspace(0, len(indices) - 1, 6)]
    rows = []
    for epoch in (0, 1, 314):
        ds.set_epoch(epoch)
        for index in selected:
            item = ds[index]
            r = ds.records[index]
            top, left, h, w = item['crop_tlhw']
            rt, rl, rh, rw = r.get('train_roi_tlhw', [0, 0, 1024, 1280])
            assert rt <= top and rl <= left and top + h <= rt + rh and left + w <= rl + rw
            assert list(item['raw'].shape) == [1, *expected_lr] and list(item['gt'].shape) == [1, height, width]
            source = np.load(Path(config['data_root']) / r['raw']['path'], mmap_mode='r')[r['frame_id']]
            region = source[top:top+h, left:left+w].astype(np.float32)
            independent = np.stack([region[y::3, x::3] for y in range(3) for x in range(3)]).mean(0)
            norm = ds.recipe['raw_normalization']
            np.testing.assert_allclose(item['raw'][0].numpy(), (independent - norm['offset']) / norm['scale'], rtol=0, atol=1e-5)
            with Image.open(Path(config['data_root']) / r['target']['path']) as image:
                truth = np.asarray(image)[top:top+h, left:left+w].astype(np.float32) / 255
            np.testing.assert_array_equal(item['gt'][0].numpy(), truth)
            repeated = ds[index]
            assert torch.equal(item['raw'], repeated['raw']) and item['crop_tlhw'] == repeated['crop_tlhw']
            rows.append({'epoch': epoch, 'sample_id': item['sample_id'], 'crop_tlhw': item['crop_tlhw']})
    # Selection and metric input must stay identical, irrespective of new training crop.
    evaluated = 0
    for split in ('val', 'test'):
        before, after = dataset_for_config(old_config, split), dataset_for_config(config, split)
        assert before.records == after.records
        for index in range(len(before)):
            a, b = before[index], after[index]
            assert a['crop_tlhw'] == b['crop_tlhw']
            assert torch.equal(a['raw'], b['raw']) and torch.equal(a['gt'], b['gt'])
            evaluated += 1
    rejected = []
    for split, crop in [('train', [1152, 1152]), ('train', [769, 768]), ('val', [768, 768])]:
        try:
            RawDisplayDataset(config['data_root'], split, dataset_version='dataset_d1',
                              scene_ids=config['scene_ids'], train_crop_hr=crop)
        except ValueError:
            rejected.append([split, crop])
        else:
            raise AssertionError('Invalid override was accepted')
    result = {'status': 'passed', 'data_lock_sha256': lock,
              'selected_data_sha256': record_selected_data(config, args.output),
              'train_counts': dict(Counter(r['scene_id'] for r in ds.records)),
              'train_crop_lr': expected_lr, 'train_crop_hr': [height, width],
              'all_train_regions_checked': len(ds), 'independently_rebuilt_pairs': len(rows),
              'unchanged_eval_pairs': evaluated, 'invalid_overrides_rejected': rejected,
              'padding': False, 'resizing_gt': False, 'audited_crops': rows}
    atomic_json(args.output / 'geometry.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'audited_crops'}), flush=True)


if __name__ == '__main__':
    main()
