"""Compare actual 9/7/5/3 channel inputs across seven complete sequences."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw
from run_motion import (ROOT, R2, V13, load_group, night_source, ShortHistory,
                        NightFixed, BoardHalfOptimized, structural, measure)
from candidate_ops import HistoryGate as ExistingGate
from gap_metrics_v2 import GapMetrics


class HistoryGate(ExistingGate):
    def __init__(self, threshold, strength=.75, count=5):
        super().__init__(threshold, strength, 3)
        if count not in [3, 5, 7, 9]:
            raise ValueError('Unsupported input count')
        self.count = count
        self.detector = 'peak'


def fold_conv(old, count):
    assert old.in_channels == 9 and old.groups == 1
    new = nn.Conv2d(count, old.out_channels, old.kernel_size, old.stride, old.padding,
                    dilation=old.dilation, bias=old.bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        # Balanced contiguous groups preserve the temporal weight sum for static input.
        for channel in range(9):
            new.weight[:, min(count-1, channel*count//9)].add_(old.weight[:, channel])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class GatedBoard(nn.Module):
    def __init__(self, board, gate):
        super().__init__()
        self.board = board
        self.gate = gate

    def forward(self, raw, context):
        return self.board(self.gate(raw), context)


def make_board(source, count, box, kind=None):
    shortened = copy.deepcopy(source)
    if kind:
        if count < 9:
            shortened.front.temporal_conv = fold_conv(shortened.front.temporal_conv, count)
        board = NightFixed(shortened, kind).cuda().eval()
        attr = 'output'
    else:
        if count < 9:
            shortened.quarter.front.first = fold_conv(shortened.quarter.front.first, count)
        shortened.current_only = False
        board = BoardHalfOptimized(shortened, box).cuda().half().eval()
        attr = 'rows'
    structural.replace_samples(board, 8)
    setattr(board, attr, structural.ChannelCopyOutput().cuda().half())
    return board


def read_gt(config, row, frame):
    path = (Path(config['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')
    return np.asarray(Image.open(path), dtype=np.float32)


def predict(model, args):
    with torch.inference_mode():
        image = model(*args)
        assert image.shape == (1, 1, 3072, 3840) and image.dtype == torch.float16
        assert torch.isfinite(image).all()
        return image[0, 0, ::3, ::3].float().cpu().numpy()


def run_scene(scene, out, group=None, steps=200, timing_enabled=False):
    kind = scene.removeprefix('night_') if scene.startswith('night_') else None
    if kind:
        source = night_source(kind)
        from ir_sr.training import dataset_for_config
        manifest = json.loads((V13/'night_calibration_final'/scene/'manifest.json').read_text())
        config = manifest['config']
        df = dataset_for_config
        caldir = V13/'night_calibration_final'/scene
        box = None
    else:
        _, source, config, _, _, df, box_for = load_group(group)
        caldir = V13/'deployment_corrected'/scene
    data = df(config, 'test')
    row = next(r for r in data.records if r['scene_id'] == scene)
    if not kind:
        box = box_for(row).cuda()
    models = {}
    counts = {}
    thresholds = {}
    for count in [9, 7, 5]:
        name = f'original9' if count == 9 else f'fold{count}'
        models[name] = make_board(source, count, box, kind)
        counts[name] = count
    if not kind:
        trained = ShortHistory(source, 3, 'half')
        checkpoint = R2/f'half3_{group}_3/best.pt'
        trained.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        board = BoardHalfOptimized(trained.source, box).cuda().half().eval()
        structural.replace_samples(board, 8)
        board.rows = structural.ChannelCopyOutput().cuda().half()
        models['trained3'] = board
        counts['trained3'] = 3
    # Only train RAW supplies gate thresholds. No validation/test threshold fitting.
    for name in ['fold5', 'original9' if kind else 'trained3']:
        count = counts[name]
        probe = HistoryGate(1, count=count).cuda()
        signals = []
        with torch.inference_mode():
            for path in sorted(caldir.glob('train_*.npz')):
                with np.load(path, allow_pickle=False) as a:
                    raw = torch.from_numpy(a['nine_raw'][:, -count:]).cuda().float()
                signals.append(probe.signal(raw)[:, :, 3:-3, 3:-3].cpu().numpy().ravel())
        assert len(signals) == 4
        threshold = max(float(np.percentile(np.concatenate(signals), 95)), 1e-6)
        label = name+'_peak'
        thresholds[label] = threshold
        models[label] = GatedBoard(models[name], HistoryGate(threshold, .75, count).cuda()).eval()
        counts[label] = count
    def inputs(dataset, rr):
        raw = dataset.normalized_stack(rr, (0, 0, 1024, 1280))[None].cuda()
        c, _ = dataset.context_for(rr, **({'crop_tlhw': (0, 0, 1024, 1280)} if kind else {}))
        c = c[None].cuda()
        if not kind:
            raw, c = raw.half(), c.half()
        elif kind == 'ordinary':
            raw = raw.half()
        return raw, c
    dest = out/scene
    dest.mkdir()
    # Each validation record is paired with its true previous GT frame.
    val = df(config, 'val')
    vrows = [r for r in val.records if r['scene_id'] == scene][:8]
    assert vrows
    validation_background = GapMetrics(vrows[0]['eval_crop_tlhw'], models)
    validation_background.prepare_background([read_gt(config, r, int(r['frame_id'])) for r in vrows])
    val_reports = []
    for rr in vrows:
        frame = int(rr['frame_id'])
        metric = GapMetrics(rr['eval_crop_tlhw'], models)
        metric.background = validation_background.background
        metric.noise = validation_background.noise
        for f in sorted(set([max(0, frame-1), frame])):
            r = dict(rr, frame_id=f)
            raw, c = inputs(val, r)
            preds = {name: predict(model, (raw[:, -counts[name]:], c)) for name, model in models.items()}
            metric.update(f, read_gt(config, rr, f), preds)
        val_reports.append(metric.report())
    validation = {}
    for name in models:
        psnr = [r['frames'][-1]['methods'][name]['psnr_db'] for r in val_reports]
        targets = [t['methods'][name] for r in val_reports for t in r['targets']]
        validation[name] = {'psnr_db': float(np.mean(psnr)), 'target_observations': len(targets),
                            'gap_error_gray': float(np.mean([t['gap_error_gray'] for t in targets])) if targets else None,
                            'gap_ratio_to_gt': float(np.mean([t['gap_ratio_to_gt'] for t in targets])) if targets else None}
    from ir_sr.sequence_normalization import regional_key
    norm = data.base.sequence_normalization
    key = regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    count = int(norm.by_key[key]['frames'])
    metric = GapMetrics(row['eval_crop_tlhw'], models)
    metric.prepare_background([read_gt(config, row, f) for f in range(count)])
    histories = []
    for frame in range(count):
        rr = dict(row, frame_id=frame)
        ids = [int(v) for v in data.frame_ids(rr)]
        assert max(ids) <= frame
        histories.append({str(k): ids[-k:] for k in [9, 7, 5, 3]})
        raw, c = inputs(data, rr)
        preds = {name: predict(model, (raw[:, -counts[name]:], c)) for name, model in models.items()}
        metric.update(frame, read_gt(config, row, frame), preds)
        if frame%30 == 0:
            print('REDUCED_FIELD', scene, frame, len(metric.targets), flush=True)
    report = metric.report()
    prepared = []
    for path in sorted(caldir.glob('test_*.npz')):
        with np.load(path, allow_pickle=False) as a:
            args = tuple(torch.from_numpy(a[k]).cuda() for k in ['nine_raw', 'reference_thumb'])
        if not kind:
            args = tuple(v.half() for v in args)
        prepared.append(args)
    assert len(prepared) == 12
    timing = {}
    # Full outputs. Each model receives K actual channels. Input slicing is outside timing.
    for name, model in (models.items() if timing_enabled else []):
        args = [(raw[:, -counts[name]:].contiguous(), c) for raw, c in prepared]
        timing[name] = measure(model, args, 50, steps)
    reference = 'original9' if kind else 'trained3'
    report.update(scene=scene, frames_count=count, validation=validation, timing=timing,
                  actual_input_channels=counts, history_ids=histories, gate_training_thresholds=thresholds,
                  deployed_reference=reference, reference_thumb='current frame only',
                  timing_note='same model timing from reduced_<group> run; not repeated' if not timing_enabled else 'new full output measurement',
                  automatically_adopted=False, NPU_measured=False, SDK_compiled=False,
                  reduction='balanced temporal weight folding; static temporal sum preserved; no fine tuning yet')
    (dest/'full_field_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if metric.worst:
        worst = metric.worst
        images = worst['images']
        tiles = []
        for name, a in images.items():
            tile = Image.fromarray(np.rint(a.clip(0, 255)).astype(np.uint8)).convert('RGB').resize((256, 256), Image.Resampling.NEAREST)
            canvas = Image.new('RGB', (256, 288), '#181818')
            canvas.paste(tile, (0, 32))
            ImageDraw.Draw(canvas).text((5, 8), name, fill='white')
            tiles.append(canvas)
        sheet = Image.new('RGB', (256*len(tiles), 288))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (256*i, 0))
        sheet.save(dest/'weak_gap_example.png')
        (dest/'weak_gap_example.json').write_text(json.dumps(worst['record'], indent=2))
    summary = {'scene': scene, 'frames': count, 'target_observations': report['target_observations'],
               'validation': validation, 'summary': report['summary'], 'actual_input_channels': counts,
               'timing': {n: {k: {j: v for j, v in m.items() if j in ['mean_ms', 'p95_ms']} for k, m in t.items() if isinstance(m, dict)} for n, t in timing.items()}}
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    print('REDUCED_COMPLETE', scene, json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--timing', action='store_true')
    p.add_argument('--group', choices=['day', 'light', 'heavy', 'night', 'all'], default='all')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    specs = [('day', ['day_normal']), ('light', ['weather_light', 'weather_medium']),
             ('heavy', ['weather_heavy', 'weather_heavy_c32']), ('night', ['night_ordinary', 'night_special'])]
    for group, scenes in specs:
        if a.group not in ['all', group]:
            continue
        for scene in scenes:
            run_scene(scene, a.out, None if group == 'night' else group, a.steps, a.timing)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

