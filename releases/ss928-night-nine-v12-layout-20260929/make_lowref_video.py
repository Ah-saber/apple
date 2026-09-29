"""Render full-field night input, GT, safe model and trained reference shortcut."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from lowres_reference import load_candidate as load_lowref, prepare_inputs
from output_candidates import load_candidate as load_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--v09-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root / 'code/worktrees/'
        'ss928-quality-special-night-nine-frame-20260925/src'))
    from ir_sr.training import dataset_for_config
    state = torch.load(args.root / 'runs/'
        'SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1/'
        'checkpoints/step_000002000.pt', map_location='cpu', weights_only=False)
    dataset = dataset_for_config(state['config'], 'test')
    record = next(r for r in dataset.records if r['scene_id'] == 'night_special')
    with (args.v09_run.parent / 'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(
            2.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        models = {'safe': load_safe('special', 'rows32', args.v09_run),
                  'lowref_k5': load_lowref('special', 5, 'rows32',
                                          args.v09_run, trained=True)}
        video = args.out / 'special_lowref_k5_full_comparison.mp4'
        writer = subprocess.Popen([
            'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '5120x1080', '-r', '12', '-i', '-', '-an', '-c:v', 'libx264',
            '-threads', '4', '-preset', 'veryfast', '-crf', '18', '-pix_fmt',
            'yuv420p', '-movflags', '+faststart', '-y', str(video)],
            stdin=subprocess.PIPE)
        font = ImageFont.truetype(
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 27)
        differences = []
        try:
            with torch.inference_mode():
                for frame in range(120):
                    row = dict(record, frame_id=frame)
                    raw = dataset.normalized_stack(row,
                        (0, 0, 1024, 1280))[None].cuda()
                    context, _ = dataset.context_for(row,
                        crop_tlhw=(0, 0, 1024, 1280))
                    context = context[None].cuda()
                    images = {}
                    for name, model in models.items():
                        x, c = prepare_inputs(model, raw, context)
                        value = model(x, c).float().clamp(0, 255).round()
                        assert tuple(value.shape) == (1, 1, 3072, 3840)
                        images[name] = F.avg_pool2d(value, 3, 3)[0, 0].cpu().numpy()
                        if frame == 60:
                            Image.fromarray(value[0, 0].byte().cpu().numpy()).save(
                                args.out / f'special_{name}_native3x_frame_060.png')
                    target = (Path(state['config']['data_root']) /
                        row['target']['path']).with_name(f'{frame:06d}.png')
                    gt = np.asarray(Image.open(target), dtype=np.float32)
                    norm = dataset.normalization_for(row)
                    original = ((dataset.base._raw(row).astype(np.float32) -
                                 norm['offset']) / norm['scale'] * 255)
                    canvas = Image.new('RGB', (5120, 1080), '#17191c')
                    draw = ImageDraw.Draw(canvas)
                    labels = ('输入 RAW', 'GT', '当前安全版', '夜间低分辨率参考 k5')
                    values = (original, gt, images['safe'], images['lowref_k5'])
                    for index, (label, value) in enumerate(zip(labels, values)):
                        assert value.shape == (1024, 1280)
                        draw.text((index * 1280 + 18, 12),
                            f'{label} 第{frame:03d}帧', font=font, fill='white')
                        canvas.paste(Image.fromarray(np.rint(np.clip(
                            value, 0, 255)).astype(np.uint8)).convert('RGB'),
                            (index * 1280, 56))
                    writer.stdin.write(np.asarray(canvas).tobytes())
                    differences.append(float(np.abs(images['lowref_k5'] -
                                                    images['safe']).mean()))
                    if frame in (0, 60, 119):
                        canvas.save(args.out / f'special_lowref_k5_frame_{frame:03d}.jpg',
                                    quality=95)
                    if frame % 10 == 0:
                        print('VIDEO', frame, flush=True)
        finally:
            writer.stdin.close()
            assert writer.wait() == 0
        metadata = {'scene': 'special', 'variant': 'lowref_k5_rows32',
            'count': 120, 'fps': 12, 'video_shape': [1080, 5120],
            'each_panel_sensor_shape': [1024, 1280],
            'model_output_shape': [1, 1, 3072, 3840],
            'display': 'full field, output averaged per 3x3 sensor cell',
            'source_execution': True, 'NPU_verified': False,
            'mean_rawgrid_difference_gray': float(np.mean(differences))}
        (args.out / 'special_lowref_k5_video.json').write_text(
            json.dumps(metadata, indent=2))
        print('FINISHED', video, flush=True)


if __name__ == '__main__':
    main()
