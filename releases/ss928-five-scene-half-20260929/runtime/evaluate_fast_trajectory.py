"""Evaluate the pre-existing fast trajectory statistic on frozen trained weights."""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import torch


PREFIX = 'SS928-FIVE-SCENE-NINE-SCRATCH-20260929-'
RUNS = ('DAY-FULLNIGHT', 'LIGHT-FULLNIGHT', 'HEAVY-FULLNIGHT', 'C32-FULLNIGHT')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), required=True)
    p.add_argument('--runs', nargs='+', choices=RUNS, default=RUNS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.code / 'src'), str(a.runtime)]
    from ir_sr.model import inference_model
    from ir_sr.training import evaluate
    from trajectory_batched_stats import trajectory_features_batched_stats

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    report = {'split': a.split, 'trajectory': 'prior-night-half-resolution-trimmed',
              'results': {}}
    for name in a.runs:
        checkpoint = a.root / 'runs' / f'{PREFIX}{name}-20K/checkpoints/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = state['config']
        model = inference_model(config, state['model']).eval().cuda()
        model._trajectory_features = types.MethodType(trajectory_features_batched_stats, model)
        result = evaluate(model, config['data_root'], a.split, 'cuda',
                          a.out / name, state['progress']['step'],
                          save_examples=False, scene_ids=config['scene_ids'], config=config)
        report['results'][name] = {
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'selected_step': state['progress']['step'],
            'original_validation_best_psnr': state['best_val_macro_psnr'],
            'macro': result['macro'], 'scene_metrics': result['scene_metrics'],
        }
        print(name, json.dumps(report['results'][name], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
