"""Verify real-data crop/augmentation correspondence without modifying frozen D1."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image, ImageDraw
import torch
from ir_sr.data import geometric_transform
from ir_sr.training import dataset_for_config, atomic_json, validate_data_lock
from ir_sr.auxiliary import validate_middle_lock


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); config = json.loads(args.config.read_text())
    validate_data_lock(config['data_root'], Path(__file__).resolve().parents[1])
    augmented = dataset_for_config(config, 'train', use_cache=True)
    middle_sha = validate_middle_lock(config, augmented)
    plain_config = dict(config, geometric_augmentation=False)
    plain = dataset_for_config(plain_config, 'train', use_cache=True)
    seen = Counter(); records = []
    selected = []
    for scene in config['scene_ids']:
        indices = [i for i,r in enumerate(augmented.records) if r['scene_id']==scene]
        selected += [indices[i] for i in np.linspace(0,len(indices)-1,min(32,len(indices)),dtype=int)]
    for epoch in (0, 1):
        augmented.set_epoch(epoch); plain.set_epoch(epoch)
        for i in selected:
            a, b = augmented[i], plain[i]
            assert a['crop_tlhw'] == b['crop_tlhw']
            op = a['augmentation_id']; seen[op] += 1
            for key in ('raw','gt','middle_raw'):
                if key in b:
                    expected = torch.from_numpy(geometric_transform(b[key].numpy(),op))
                    torch.testing.assert_close(a[key],expected,rtol=0,atol=0)
            assert a['raw'].shape == (1,256,256) and a['gt'].shape == (1,768,768)
            assert all(torch.isfinite(a[k]).all() for k in ('raw','gt'))
            records.append({'sample_id':a['sample_id'],'epoch':epoch,'crop_tlhw':a['crop_tlhw'],'augmentation_id':op})
    assert set(seen) == set(range(8)), seen
    evaluation = {}
    for split in ('val','test'):
        a = dataset_for_config(config, split); b = dataset_for_config(plain_config,split)
        for i in range(len(a)):
            x,y = a[i],b[i]
            assert x['augmentation_id']==0 and 'middle_raw' not in x
            torch.testing.assert_close(x['raw'],y['raw'],rtol=0,atol=0)
            torch.testing.assert_close(x['gt'],y['gt'],rtol=0,atol=0)
        evaluation[split] = len(a)
    plain.set_epoch(0); example = plain[0]
    columns = ['raw','gt'] + (['middle_raw'] if 'middle_raw' in example else [])
    canvas = Image.new('RGB',(256*len(columns),8*282),'#202020'); draw=ImageDraw.Draw(canvas)
    for op in range(8):
        for col,key in enumerate(columns):
            a = geometric_transform(example[key][0].numpy(),op)
            a = a if key=='gt' else (a+3)/6
            im = Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
            canvas.paste(im,(col*256,op*282+26))
            draw.text((col*256+4,op*282+6),f'op {op}: {key}',fill='white')
    canvas.save(args.output/'augmentation_contact_sheet.png')
    result = {'status':'passed','train_counts':dict(Counter(r['scene_id'] for r in augmented.records)),
              'actual_sample_epoch_checks':len(records),'operations':dict(seen),'fixed_evaluation_counts':evaluation,
              'middle_gt_index_sha256':middle_sha,'records':records,'wall_seconds':time.perf_counter()-start,
              'visual_source_sample':example['sample_id'],'protocol':'exact synchronized D4 on already paired crops; no interpolation; fixed validation/test'}
    atomic_json(args.output/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
