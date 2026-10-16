"""Train only the temporal/spatial input front using GT pair and local gap losses."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from run_reduced_v2 import (ROOT, R2, V13, load_group, ShortHistory, fold_conv,
                            BoardHalfOptimized, structural, measure, read_gt)
from gap_metrics_v3 import GapMetrics
from run_motion import GROUPS
from ghost_objective import departure_loss
from gt_tracks import save_gallery
from innovation_gate import InnovationConv


class ReducedHalf(nn.Module):
    def __init__(self, base, count):
        super().__init__()
        self.source = copy.deepcopy(base)
        self.count = count
        self.source.quarter.front.first = fold_conv(self.source.quarter.front.first, count)
        self.source.current_only = False

    def native(self, raw, context, box):
        return self.source.native(raw[:, -self.count:], context, box)


def gap_loss(pred, gt, teacher):
    # Training targets are causal GT pairs. No model output defines the motion masks.
    e = (pred-gt)[:, :, 12:-12, 12:-12]*255
    delta = (gt[1::2]-gt[::2])*255
    delta = F.avg_pool2d(delta, 3, 1, 1)[:, :, 12:-12, 12:-12]
    center = delta.flatten(1).median(1).values[:, None, None, None]
    sigma = 1.4826*(delta-center).abs().flatten(1).median(1).values[:, None, None, None]
    mask = F.max_pool2d(((delta-center).abs() >= torch.maximum(sigma*2.5, torch.full_like(sigma, 3))).float(), 3, 1, 1)
    weight = mask.repeat_interleave(2, 0)
    moving = (e.abs()*weight).sum()/weight.sum().clamp_min(1)
    pairs = e.reshape(-1, 2, 1, e.shape[-2], e.shape[-1])
    temporal = ((pairs[:, 1]-pairs[:, 0]).abs()*mask).sum()/mask.sum().clamp_min(1)
    pred_gap = pred-F.avg_pool2d(pred, 7, 1, 3)
    gt_gap = gt-F.avg_pool2d(gt, 7, 1, 3)
    gap_error = (pred_gap-gt_gap)[:, :, 12:-12, 12:-12].abs()*255
    gap = (gap_error*weight).sum()/weight.sum().clamp_min(1)
    departed, departure_fraction = departure_loss(pred[1::2], gt[1::2], gt[::2], mask, 12)
    static = 1-weight
    imitate = ((pred-teacher)[:, :, 12:-12, 12:-12].abs()*static).sum()/static.sum().clamp_min(1)*255
    weak = ((gt_gap.abs()*255 >= 1)&(gt_gap.abs()*255 < 3))[:, :, 12:-12, 12:-12].float()
    weak_loss = (gap_error*weak).sum()/weak.sum().clamp_min(1)
    loss = e.abs().mean()+2*moving+2*temporal+3*gap+4*departed+.25*weak_loss+.1*imitate
    return loss, {'mae_gray': float(e.detach().abs().mean()), 'moving_gray': float(moving.detach()),
                  'temporal_gray': float(temporal.detach()), 'gap_gray': float(gap.detach()),
                  'motion_fraction': float(mask.detach().mean()), 'departure_gray': float(departed.detach()),
                  'departure_fraction': departure_fraction}


def validation(models, config, data, scenes, box_for):
    result = {}
    with torch.inference_mode():
        for scene in scenes:
            rows = [r for r in data.records if r['scene_id'] == scene][:8]
            assert rows
            template = GapMetrics(rows[0]['eval_crop_tlhw'], models)
            template.prepare_background([read_gt(config, r, int(r['frame_id'])) for r in rows])
            observations = []
            for row in rows:
                metric = GapMetrics(row['eval_crop_tlhw'], models)
                metric.background, metric.noise = template.background, template.noise
                frame = int(row['frame_id'])
                for f in range(max(0, frame-5), frame+1):
                    rr = dict(row, frame_id=f)
                    raw = data.normalized_stack(rr, (0, 0, 1024, 1280))[None].cuda()
                    context, _ = data.context_for(rr)
                    context = context[None].cuda()
                    box = box_for(rr).cuda()
                    ps = {name: (m.native(raw, context, box)[0, 0].float().clamp(0, 1)*255).cpu().numpy()
                          for name, m in models.items()}
                    metric.update(f, read_gt(config, row, f), ps)
                observations.append(metric.report())
            result[scene] = {}
            for name in models:
                frames = [r['frames'][-1]['methods'][name] for r in observations]
                targets = [t['methods'][name] for r in observations for t in r['targets']]
                keys = set().union(*(f.keys() for f in frames))
                frame_metrics = {k: float(np.mean([f[k] for f in frames if k in f])) for k in keys}
                target_metrics = {k: float(np.mean([t[k] for t in targets])) for k in targets[0]} if targets else {}
                result[scene][name] = {'frames': frame_metrics, 'targets': target_metrics, 'target_observations': len(targets)}
    return result


def run(group, count, out, steps):
    _, base, config, _, _, df, box_for = load_group(group)
    base = base.cuda().float().eval()
    reference = ShortHistory(base, 3, 'half').cuda().float().eval()
    reference.load_state_dict(torch.load(R2/f'half3_{group}_3/best.pt', map_location='cpu', weights_only=False)['model'])
    model = ReducedHalf(base, count).cuda().float()
    if count == 3:
        model.source.load_state_dict(reference.source.state_dict())
    initial = Path(__file__).resolve().parents[1]/('ghost_v3_'+group+'5')/'best.pt'
    if initial.exists():
        checkpoint = torch.load(initial, map_location='cpu', weights_only=False)
        if checkpoint['count'] == count:
            model.load_state_dict(checkpoint['model'])
            print('INNOVATION_INIT', str(initial), checkpoint['step'], flush=True)
    model.requires_grad_(False)
    first = InnovationConv(model.source.quarter.front.first)
    model.source.quarter.front.first = first
    first.gate.requires_grad_(True)
    reference.requires_grad_(False)
    cache = ROOT/'runs'/f'SS928-FIVE-SCENE-{GROUPS[group]["cache"]}-20260929/train_cache.pt'
    saved = torch.load(cache, map_location='cpu', weights_only=False)
    values, metadata = saved['values'], saved['metadata']
    first.calibrate([values['stack'][:8].flatten(0, 1)[:, -count:].cuda().float()])
    assert all(r['scene'] in GROUPS[group]['scenes'] for r in metadata)
    rng = np.random.default_rng(1008)
    torch.manual_seed(1008)
    val = df(config, 'val')
    scenes = GROUPS[group]['scenes']
    reference_metrics = validation({'reference3': reference}, config, val, scenes, box_for)
    best = float('inf')
    logs = []
    def validate(step):
        nonlocal best
        result = validation({'candidate': model.eval()}, config, val, scenes, box_for)
        eligible = True
        scores = []
        for scene in scenes:
            candidate = result[scene]['candidate']
            control = reference_metrics[scene]['reference3']
            c, r = candidate['frames'], control['frames']
            if c['psnr_db'] < r['psnr_db']-.05:
                eligible = False
            if c.get('static_temporal_error_gray', 0) > r.get('static_temporal_error_gray', float('inf'))*1.1+.02:
                eligible = False
            if c.get('weak_structure_error_gray', 0) > r.get('weak_structure_error_gray', float('inf'))*1.05+.02:
                eligible = False
            if c.get('old_position_signed_ghost_gray', 0) > r.get('old_position_signed_ghost_gray', float('inf'))+.02:
                eligible = False
            if candidate['targets'] and control['targets']:
                if candidate['targets']['gap_error_gray'] > control['targets']['gap_error_gray']+.02:
                    eligible = False
            score = c['mae_gray']+.25*c.get('weak_structure_error_gray', 0)+.25*c.get('old_position_signed_ghost_gray', 0)
            if candidate['targets']:
                score += .5*candidate['targets']['gap_error_gray']
            scores.append(score)
        value = float(np.mean(scores))
        ck = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
              'group': group, 'count': count, 'step': step, 'validation': result,
              'reference_validation': reference_metrics, 'eligible': eligible, 'score': value,
              'selection': 'validation only; PSNR loss <=.05dB and static temporal guard; MAE, target gap, ghost, weak structure score'}
        torch.save(ck, out/'last.pt')
        if eligible and value < best:
            best = value
            torch.save(ck, out/'best.pt')
        print('GAP_VALIDATION', group, count, step, eligible, value, flush=True)
        return {'step': step, 'eligible': eligible, 'score': value, 'metrics': result}
    logs.append(validate(0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-6)
    started = time.perf_counter()
    for step in range(1, steps+1):
        ids = rng.choice(len(metadata), 4)
        raw = values['stack'][ids].flatten(0, 1).cuda().float()
        context = values['context'][ids].flatten(0, 1).cuda().float()
        box = values['box'][ids].flatten(0, 1).cuda().float()
        gt = values['gt'][ids].flatten(0, 1).cuda().float()
        teacher = values['teacher'][ids].flatten(0, 1).cuda().float()
        model.train()
        with torch.autocast('cuda', enabled=False):
            pred = model.native(raw, context, box).float().clamp(0, 1)
            loss, metrics = gap_loss(pred, gt, teacher)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        schedule.step()
        if step%500 == 0 or step == steps:
            record = {'step': step, 'loss': float(loss.detach()), **metrics, 'elapsed_seconds': time.perf_counter()-started}
            record['validation'] = validate(step)
            logs.append(record)
            (out/'training.json').write_text(json.dumps({'group': group, 'count': count, 'steps': steps,
                'cache': str(cache), 'logs': logs, 'trainable': 'four local current-feature gate parameters only; all source kernels, biases, body and reference frozen',
                'loss': 'GT pixel +2 motion +2 paired temporal +3 signed gap +4 departure ghost +.25 weak +.1 static teacher',
                'initial_checkpoint': str(initial) if initial.exists() else None,
                'evaluation': 'GT background and coherent moving tracks over the full valid field',
                'training_precision': 'FP32; TF32 disabled; avoid BF16 gray-gap rounding during training',
                'hash_checks': False, 'packaged': False, 'pushed': False}, indent=2))
            print('GAP_TRAIN', group, count, step, float(loss.detach()), flush=True)
    path = out/'best.pt'
    if not path.exists():
        (out/'selection.json').write_text(json.dumps({'selected': False, 'reason': 'No validation candidate passes PSNR/static guards'}))
        return
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    test = df(config, 'test')
    for scene in scenes:
        dest = out/scene
        dest.mkdir()
        row = next(r for r in test.records if r['scene_id'] == scene)
        box = box_for(row).cuda()
        boards = {}
        for name, src in [('reference3', reference.source), ('gap_candidate', model.source)]:
            board = BoardHalfOptimized(src, box).cuda().half().eval()
            structural.replace_samples(board, 8)
            board.rows = structural.ChannelCopyOutput().cuda().half()
            boards[name] = board
        counts = {'reference3': 3, 'gap_candidate': count}
        from ir_sr.sequence_normalization import regional_key
        norm = test.base.sequence_normalization
        key = regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
        nframes = int(norm.by_key[key]['frames'])
        metric = GapMetrics(row['eval_crop_tlhw'], boards)
        metric.prepare_background([read_gt(config, row, f) for f in range(nframes)])
        with torch.inference_mode():
            for frame in range(nframes):
                rr = dict(row, frame_id=frame)
                raw = test.normalized_stack(rr, (0, 0, 1024, 1280))[None].cuda().half()
                context, _ = test.context_for(rr)
                context = context[None].cuda().half()
                predictions = {name: board(raw[:, -counts[name]:], context)[0, 0, ::3, ::3].float().cpu().numpy()
                               for name, board in boards.items()}
                metric.update(frame, read_gt(config, row, frame), predictions)
        report = metric.report()
        save_gallery(metric, dest)
        prepared = []
        for p in sorted((V13/'deployment_corrected'/scene).glob('test_*.npz')):
            with np.load(p, allow_pickle=False) as a:
                prepared.append(tuple(torch.from_numpy(a[k]).cuda().half() for k in ['nine_raw', 'reference_thumb']))
        timing = {name: measure(board, [(x[:, -counts[name]:].contiguous(), c) for x, c in prepared], 50, 300)
                  for name, board in boards.items()}
        report.update(scene=scene, checkpoint=str(path), validation_selected_step=ck['step'],
                      actual_input_channels=counts, timing=timing, automatically_adopted=False, NPU_measured=False)
        (dest/'full_field_results.json').write_text(json.dumps(report, indent=2))
        summary = {k: report[k] for k in ['scene', 'target_observations', 'summary', 'actual_input_channels', 'validation_selected_step']}
        summary['timing'] = {n: {k: {j: v for j, v in m.items() if j in ['mean_ms', 'p95_ms']} for k, m in t.items()} for n, t in timing.items()}
        (dest/'summary.json').write_text(json.dumps(summary, indent=2))
        print('GAP_COMPLETE', scene, count, json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['day', 'light', 'heavy'], required=True)
    p.add_argument('--count', type=int, choices=[3, 5, 7], default=5)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.group, a.count, a.out, a.steps)


if __name__ == '__main__':
    main()


