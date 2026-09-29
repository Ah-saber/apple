"""Fit the four-channel low-resolution reference on training contexts only."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lowres_reference import load_candidate, tail_conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--kernel', type=int, choices=[1, 3, 5], required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        special = args.scene == 'special'
        work = ('ss928-quality-special-night-nine-frame-20260925' if special
                else 'ss928-quality-night-nine-frame-20260925')
        run = ('SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special
               else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1')
        sys.path.insert(0, str(args.root / 'code/worktrees' / work / 'src'))
        from ir_sr.training import dataset_for_config
        state = torch.load(args.root / 'runs' / run /
            'checkpoints/step_000002000.pt', map_location='cpu',
            weights_only=False)
        dataset = dataset_for_config(state['config'], 'train')
        records = [r for r in dataset.records
                   if r['scene_id'] == 'night_' + args.scene]
        assert len(records) > 40
        train_records, held_records = records[:-8], records[-8:]
        contexts = []
        for record in records:
            context, _ = dataset.context_for(record)
            contexts.append(context.numpy().astype(np.float32))
        values = torch.from_numpy(np.stack(contexts)).cuda()
        model = load_candidate(args.scene, args.kernel, 'rows32',
                               args.v09_run).eval()
        model.requires_grad_(False)
        model.low_ref = model.low_ref.float()
        model.low_ref.requires_grad_(True)
        tail = tail_conv(model.core.model.tail)
        projection = model.output.conv
        optimizer = torch.optim.AdamW(model.low_ref.parameters(), lr=.001,
                                      weight_decay=1e-6)
        torch.manual_seed(928 + args.kernel + int(special) * 100)
        logs = []
        best = float('inf')
        best_step = 0
        started = time.perf_counter()

        def pair(context):
            with torch.no_grad():
                reference = model.reference_features(context).float()
                full = F.interpolate(reference, size=(512, 640),
                                     mode='bilinear', align_corners=False)
                stage = F.conv2d(full, tail.weight.float(), padding=1)
                teacher = F.conv2d(stage, projection.weight.float(),
                                   padding=1)
            prediction = F.interpolate(model.low_ref(reference),
                size=(512, 640), mode='bilinear', align_corners=False)
            return prediction, teacher

        def held_metric():
            total = 0.
            count = 0
            with torch.no_grad():
                for context in values[len(train_records):].split(2):
                    output, target = pair(context)
                    total += float((output - target).square().sum())
                    count += output.numel()
            return (total / count) ** .5

        initial = held_metric()
        print('INITIAL_HELD_RMSE', args.scene, args.kernel, initial, flush=True)
        for step in range(1, args.steps + 1):
            indices = torch.randint(0, len(train_records), (2,), device='cuda')
            context = values.index_select(0, indices)
            context = context * (.8 + .4 * torch.rand(2, 1, 1, 1,
                device='cuda')) + (torch.rand(2, 1, 1, 1,
                device='cuda') - .5) * .3
            output, target = pair(context)
            loss = (output - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == 1:
                held = held_metric()
                row = {'step': step, 'train_mse': float(loss),
                       'held_rmse_phase': held}
                logs.append(row)
                print('TRAIN', args.scene, args.kernel, row, flush=True)
                if held < best:
                    best, best_step = held, step
                    checkpoint = {'scene': args.scene, 'kernel': args.kernel,
                        'step': step, 'held_rmse_phase': held,
                        'train_context_count': len(train_records),
                        'held_context_count': len(held_records),
                        'test_used': False,
                        'low_ref': {key: value.detach().cpu().half().clone()
                                    for key, value in
                                    model.low_ref.state_dict().items()}}
                    torch.save(checkpoint, args.out /
                        f'{args.scene}_lowref_k{args.kernel}.pt')
        report = {'scene': args.scene, 'kernel': args.kernel,
                  'steps': args.steps, 'initial_held_rmse_phase': initial,
                  'best_held_rmse_phase': best, 'best_step': best_step,
                  'elapsed_seconds': time.perf_counter() - started,
                  'train_records': [r['sample_id'] for r in train_records],
                  'held_records': [r['sample_id'] for r in held_records],
                  'test_used': False, 'logs': logs}
        (args.out / f'{args.scene}_lowref_k{args.kernel}_training.json').write_text(
            json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
