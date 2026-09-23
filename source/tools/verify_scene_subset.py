"""Check selected D1 membership, identical geometry/normalization and scene balance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from ir_sr.data import RawDisplayDataset
from ir_sr.training import dataset_for_config, record_selected_data, atomic_json, validate_data_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config = json.loads(args.config.read_text())
    lock = validate_data_lock(config['data_root'],Path(__file__).resolve().parents[1])
    checks = {}
    for split in ('train','val','test'):
        full = RawDisplayDataset(config['data_root'],split,dataset_version='dataset_d1')
        subset = dataset_for_config(config,split,use_cache=(split=='train'))
        expected = [r for r in full.records if r['scene_id'] in config['scene_ids']]
        assert subset.records == expected
        assert subset.recipe == full.recipe
        by_id = {r['sample_id']:i for i,r in enumerate(full.records)}
        chosen = list(range(len(subset))) if split!='train' else [i for i in range(0,len(subset),20)]
        for i in chosen:
            a,b=subset[i],full[by_id[subset.records[i]['sample_id']]]
            assert a['crop_tlhw']==b['crop_tlhw']
            torch.testing.assert_close(a['raw'],b['raw'],rtol=0,atol=0)
            torch.testing.assert_close(a['gt'],b['gt'],rtol=0,atol=0)
        checks[split]={'counts':dict(Counter(r['scene_id'] for r in subset.records)),
                       'tensor_parity_samples':len(chosen),'allowed_regions_unchanged':True}
        if split=='train':
            sums={s:sum(float(w) for r,w in zip(subset.records,subset.balanced_sampler().weights)
                         if r['scene_id']==s) for s in config['scene_ids']}
            assert max(sums.values())-min(sums.values())<1e-10
            checks[split]['scene_probability']={s:v/sum(sums.values()) for s,v in sums.items()}
    for invalid in ([],['night_special','night_special'],['missing_scene']):
        try:RawDisplayDataset(config['data_root'],'train',dataset_version='dataset_d1',scene_ids=invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid subset accepted')
    receipt={'status':'passed','data_lock_sha256':lock,
             'selected_data_sha256':record_selected_data(config,args.output),'checks':checks}
    atomic_json(args.output/'verification.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
