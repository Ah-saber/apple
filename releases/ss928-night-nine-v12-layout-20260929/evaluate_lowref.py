"""Compare low-resolution reference to frozen safe model on the test sequence."""
import argparse
import fcntl
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from lowres_reference import load_candidate as load_lowref, prepare_inputs
from output_candidates import load_candidate as load_output


def tiled_stats(sequence):
    medians = []
    spreads = []
    for frame in sequence:
        height, width = frame.shape
        mid = []
        spread = []
        for y in range(8):
            for x in range(8):
                cell = frame[y * height // 8:(y + 1) * height // 8,
                             x * width // 8:(x + 1) * width // 8]
                low, median, high = np.percentile(cell, (10, 50, 90))
                mid.append(median)
                spread.append(high - low)
        medians.append(mid)
        spreads.append(spread)
    return np.asarray(medians), np.asarray(spreads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['ordinary', 'special'], required=True)
    parser.add_argument('--kernels', nargs='+', type=int, default=[3, 5])
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
        torch.cuda.set_per_process_memory_fraction(
            2.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        special = args.scene == 'special'
        work = ('ss928-quality-special-night-nine-frame-20260925' if special
                else 'ss928-quality-night-nine-frame-20260925')
        run = ('SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special
               else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1')
        sys.path.insert(0, str(args.root / 'code/worktrees' / work / 'src'))
        from ir_sr.training import dataset_for_config
        from ir_sr.metrics import image_metrics
        state = torch.load(args.root / 'runs' / run /
            'checkpoints/step_000002000.pt', map_location='cpu',
            weights_only=False)
        dataset = dataset_for_config(state['config'], 'test')
        record = next(r for r in dataset.records
                      if r['scene_id'] == 'night_' + args.scene)
        y, x, h, w = record['eval_crop_tlhw']
        roi = (slice(y, y + h), slice(x, x + w))
        models = {'safe': load_output(args.scene, 'rows32', args.v09_run)}
        for kernel in args.kernels:
            models[f'lowref_k{kernel}'] = load_lowref(args.scene, kernel,
                'rows32', args.v09_run, trained=True)
        scores = {name: [] for name in models}
        ssim = {name: [] for name in models}
        series = {name: [] for name in list(models) + ['gt']}
        differences = {name: [] for name in models}
        count = 120 if special else 60
        with torch.inference_mode():
            for frame in range(count):
                row = dict(record, frame_id=frame)
                raw = dataset.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
                context, _ = dataset.context_for(row,
                    crop_tlhw=(0, 0, 1024, 1280))
                context = context[None].cuda()
                raw, context = prepare_inputs(models['safe'], raw, context)
                path = (Path(state['config']['data_root']) /
                    row['target']['path']).with_name(f'{frame:06d}.png')
                gt = np.asarray(Image.open(path), dtype=np.float32)
                series['gt'].append(gt[roi][::2, ::2].astype(np.float16))
                safe = None
                for name, model in models.items():
                    value = model(raw, context).float().clamp(0, 255).round()
                    assert tuple(value.shape) == (1, 1, 3072, 3840)
                    prediction = F.avg_pool2d(value, 3, 3)[0, 0].cpu().numpy()
                    cropped = prediction[roi]
                    truth = gt[roi]
                    error = ((cropped[3:-3, 3:-3].astype(np.float64) -
                              truth[3:-3, 3:-3]) / 255) ** 2
                    scores[name].append(-10 * math.log10(max(float(error.mean()), 1e-12)))
                    series[name].append(cropped[::2, ::2].astype(np.float16))
                    if frame % 10 == 0:
                        ssim[name].append(image_metrics(
                            torch.from_numpy(np.ascontiguousarray(cropped / 255))[None, None],
                            torch.from_numpy(np.ascontiguousarray(truth / 255))[None, None],
                            border=3)['ssim'])
                    if name == 'safe':
                        safe = prediction
                    differences[name].append(float(np.abs(prediction - safe).mean()))
                if frame % 10 == 0:
                    print('FRAME', args.scene, frame,
                        {key: round(scores[key][-1] - scores['safe'][-1], 5)
                         for key in models}, flush=True)
        arrays = {key: np.stack(value).astype(np.float32)
                  for key, value in series.items()}
        gt = arrays['gt']
        mean = gt.mean(0)
        temporal = gt.std(0)
        gradient = np.hypot(cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3) / 8,
                            cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3) / 8)
        static = (temporal <= 1) & (gradient <= 5) & (mean > 5) & (mean < 250)
        weak = (temporal > 2) & (temporal <= 8) & (mean > 5) & (mean < 250)
        static[:4] = static[-4:] = False
        static[:, :4] = static[:, -4:] = False
        assert static.any() and weak.any()
        gt_mid, gt_spread = tiled_stats(gt)
        quality = {}
        for name in models:
            value = arrays[name]
            error = value - gt
            mid, spread = tiled_stats(value)
            quality[name] = {
                'psnr_db': float(np.mean(scores[name])),
                'ssim_every10': float(np.mean(ssim[name])),
                'static_error_temporal_std_gray': float(error.std(0)[static].mean()),
                'weak_moving_mae_gray': float(np.abs(error[:, weak]).mean()),
                'brightness_error_temporal_std_gray': float(np.median(
                    (mid - gt_mid).std(0))),
                'contrast_error_temporal_std_gray': float(np.median(
                    (spread - gt_spread).std(0))),
                'mean_full_rawgrid_difference_gray': float(np.mean(differences[name]))}
            print('QUALITY', args.scene, name, quality[name], flush=True)
        report = {'scene': args.scene, 'count': count,
                  'roi': record['eval_crop_tlhw'], 'quality': quality,
                  'per_frame_psnr': scores, 'GT_native_scale_only': True,
                  'test_used_for_checkpoint_selection': False,
                  'test_used_for_architecture_assessment': True,
                  'NPU_verified': False}
        (args.out / f'{args.scene}_lowref_quality.json').write_text(
            json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
