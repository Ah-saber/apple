"""Read-only brightness/generalization diagnostics; never corrected output scores."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from ir_sr.data import RawDisplayDataset
from ir_sr.model import RT4KSRB0
from ir_sr.training import atomic_json, sha, dataset_for_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if args.output.exists():
        raise RuntimeError('Use a new diagnostic output')
    torch.set_num_threads(2)
    result = json.loads((args.run/'result.json').read_text())
    config = json.loads((args.run/'config.json').read_text())
    weight = args.run/'artifacts/b0_deploy.pt'
    assert sha(weight) == result['deployment_weight_sha256']
    model = RT4KSRB0(config['channels'],config['blocks'],deploy=True).eval()
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model'])
    report = {'selected_checkpoint': result['selected_checkpoint'], 'splits': {},
              'protocol': 'diagnostic only, no correction to predictions or official scores; '
                          'bias MSE fraction = sum(mean(error)^2) / sum(MSE); float [0,1] clip, HR border3; '
                          'train uses 12 fixed epoch0 crops/class, val uses frozen 12 views/class; geometries differ'}
    for split in ('train','val'):
        ds = dataset_for_config(config, split)
        indices = list(range(len(ds)))
        if split == 'train':
            scene_indices = defaultdict(list)
            for i,r in enumerate(ds.records): scene_indices[r['scene_id']].append(i)
            indices = [values[int(i)] for _,values in sorted(scene_indices.items())
                       for i in np.linspace(0,len(values)-1,12).round().astype(int)]
        rows, groups = [],defaultdict(list)
        for i in indices:
            item = ds[i]
            raw = item['raw'][None]
            target = item['gt'][None][...,3:-3,3:-3].double()
            pred = model(raw).clamp(0,1)[...,3:-3,3:-3].double()
            error = pred-target
            mse = float(error.square().mean())
            bias = float(error.mean())
            row = {'sample_id':item['sample_id'],'scene_id':item['scene_id'],
                   'crop_tlhw':item['crop_tlhw'],'psnr':float(-10*np.log10(max(mse,1e-12))),
                   'mse':mse,'mae':float(error.abs().mean()),'mean_error_8bit_dn':255*bias,
                   'bias_mse':bias*bias,'residual_mse_after_mean_removal':max(0,mse-bias*bias),
                   'target_mean_8bit_dn':255*float(target.mean()),'prediction_mean_8bit_dn':255*float(pred.mean()),
                   'normalized_raw_mean':float(raw.mean()),'normalized_raw_std':float(raw.std())}
            rows.append(row);groups[row['scene_id']].append(row)
        scenes={}
        keys=['psnr','mse','mae','mean_error_8bit_dn','target_mean_8bit_dn','prediction_mean_8bit_dn',
              'normalized_raw_mean','normalized_raw_std']
        for scene,items in sorted(groups.items()):
            scenes[scene]={k:float(np.mean([r[k] for r in items])) for k in keys}
            scenes[scene]['bias_mse_fraction']=sum(r['bias_mse'] for r in items)/max(1e-12,sum(r['mse'] for r in items))
            scenes[scene]['count']=len(items)
        report['splits'][split]={'scene_summary':scenes,'images':rows}
    atomic_json(args.output,report)
    print(json.dumps({s:r['scene_summary'] for s,r in report['splits'].items()}),flush=True)


if __name__ == '__main__':
    main()
