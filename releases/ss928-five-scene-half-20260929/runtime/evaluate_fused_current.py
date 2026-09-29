"""Measure current fused half-precision runtime on validation or frozen test."""
import argparse
import hashlib
import json
import sys
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
    p.add_argument('--gate-bias-offset', type=float, default=0.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(a.code / 'src'), str(a.runtime), str(Path(__file__).resolve().parent)]
    from fast_nine_inference import build_fast_model
    from ir_sr.model import inference_model
    from ir_sr.training import evaluate

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    report = {'split': a.split, 'implementation': 'fast_trajectory_fused_fp16_context_box',
              'gate_bias_offset': a.gate_bias_offset,
              'results': {}}
    for name in a.runs:
        checkpoint = a.root / 'runs' / f'{PREFIX}{name}-20K/checkpoints/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = state['config']
        model = build_fast_model(config, state['model'], inference_model).cuda().eval()
        if a.gate_bias_offset:
            with torch.no_grad():
                model.second.bias[1].add_(a.gate_bias_offset)
        result = evaluate(model, config['data_root'], a.split, 'cuda',
                          a.out / name, state['progress']['step'],
                          save_examples=False, scene_ids=config['scene_ids'], config=config)
        report['results'][name] = {'checkpoint': str(checkpoint),
                                   'checkpoint_sha256': sha(checkpoint),
                                   'selected_step': state['progress']['step'],
                                   'macro': result['macro'],
                                   'scene_metrics': result['scene_metrics']}
        print(name, json.dumps(report['results'][name], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
