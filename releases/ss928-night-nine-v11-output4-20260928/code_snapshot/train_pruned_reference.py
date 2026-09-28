"""Distill selected ordinary reference branches using training contexts only."""
import argparse
import copy
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from reference_candidates import SelectPyramid, load_board


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--v09-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--keep', choices=['02', '1'], required=True)
    p.add_argument('--objective', choices=['encoded', 'projected'], default='encoded')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        work = a.root / 'code/worktrees/ss928-quality-night-nine-frame-20260925/src'
        sys.path.insert(0, str(work))
        from ir_sr.training import dataset_for_config
        run = a.root / 'runs/SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
        state = torch.load(run / 'checkpoints/step_000002000.pt',
                           map_location='cpu', weights_only=False)
        dataset = dataset_for_config(state['config'], 'train')
        records = [r for r in dataset.records if r['scene_id'] == 'night_ordinary']
        assert len(records) > 40
        train_records, hold_records = records[:-8], records[-8:]
        contexts = []
        for record in train_records + hold_records:
            context, _ = dataset.context_for(record)
            contexts.append(context.numpy().astype(np.float32))
        values = torch.from_numpy(np.stack(contexts)).cuda()
        teacher = load_board('ordinary', 'rows32', a.v09_run).core.model.global_reference.eval()
        teacher.requires_grad_(False)
        student = SelectPyramid(copy.deepcopy(teacher), tuple(int(c) for c in a.keep)).float().cuda()
        student.requires_grad_(True)
        params = list(student.encoder.parameters())
        for index in student.keep:
            params.extend(student.pyramid[index].parameters())
        if a.objective == 'projected':
            params.extend(student.project.parameters())
        optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=1e-6)
        best = float('inf')
        best_step = 0
        logs = []
        torch.manual_seed(928)
        began = time.perf_counter()
        for step in range(1, a.steps + 1):
            ids = torch.randint(0, len(train_records), (12,), device='cuda')
            context = values.index_select(0, ids)
            gain = .8 + .4 * torch.rand(12, 1, 1, 1, device='cuda')
            offset = (torch.rand(12, 1, 1, 1, device='cuda') - .5) * .3
            context = context * gain + offset
            with torch.no_grad():
                target = teacher.encode(context.half())
                if a.objective == 'projected':
                    target = teacher.project(target + target.mean((-2, -1), keepdim=True))
                target = target.float()
            prediction = student.encode(context)
            if a.objective == 'projected':
                prediction = student.project(prediction + prediction.mean((-2, -1), keepdim=True))
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 500 == 0 or step == 1:
                with torch.no_grad():
                    hold = values[len(train_records):]
                    target = teacher.encode(hold.half())
                    if a.objective == 'projected':
                        target = teacher.project(target + target.mean((-2, -1), keepdim=True))
                    target = target.float()
                    output = student.encode(hold)
                    if a.objective == 'projected':
                        output = student.project(output + output.mean((-2, -1), keepdim=True))
                    difference = output - target
                    rmse = float(difference.square().mean().sqrt())
                    mae = float(difference.abs().mean())
                item = {'step': step, 'train_loss': float(loss),
                        'held_rmse': rmse, 'held_mae': mae}
                logs.append(item)
                print(item, flush=True)
                if rmse < best:
                    best = rmse
                    best_step = step
                    checkpoint = {'reference': {k: v.detach().cpu().half().clone()
                                                for k, v in student.state_dict().items()},
                                  'scene': 'ordinary', 'keep': a.keep, 'step': step,
                                  'held_rmse': rmse, 'objective': a.objective,
                                  'train_context_count': len(train_records),
                                  'held_context_count': len(hold_records), 'test_used': False}
                    suffix = '_projected' if a.objective == 'projected' else ''
                    torch.save(checkpoint, a.output / f'ordinary_keep{a.keep}{suffix}_reference.pt')
        report = {'scene': 'ordinary', 'keep': a.keep, 'objective': a.objective,
                  'best_step': best_step,
                  'best_held_rmse': best, 'steps': a.steps,
                  'seconds': time.perf_counter() - began,
                  'train_records': [{'sample_id': r['sample_id'], 'frame_id': r['frame_id']}
                                    for r in train_records],
                  'held_records': [{'sample_id': r['sample_id'], 'frame_id': r['frame_id']}
                                   for r in hold_records], 'logs': logs, 'test_used': False}
        suffix = '_projected' if a.objective == 'projected' else ''
        (a.output / f'ordinary_keep{a.keep}{suffix}_training.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
