"""Finalize validation-selected B0: seven-scene metrics, deployment weights and visuals."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image, ImageDraw
import torch
from ir_sr.data import RawDisplayDataset
from ir_sr.model import RT4KSRB0, to_deploy, inference_model
from ir_sr.training import atomic_json, evaluate, sha, uint8_image, utc_now, dataset_for_config


class _SpaceToDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        return torch.nn.functional.pixel_unshuffle(x, factor)

    @staticmethod
    def symbolic(graph, x, factor):
        return graph.op('SpaceToDepth', x, blocksize_i=factor)


class ExportGrayUnshuffle(torch.nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return _SpaceToDepth.apply(x, self.factor)


def make_curves(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines()]
    train = [r for r in rows if r['kind'] == 'train_log']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r['step'] for r in train], [r['loss_l1'] for r in train], label='Total loss')
    if train and 'display_loss_l1' in train[0]:
        axes[0].plot([r['step'] for r in train], [r['display_loss_l1'] for r in train], label='Display L1')
        axes[0].plot([r['step'] for r in train], [r['auxiliary_raw_weight']*r['auxiliary_raw_loss_l1'] for r in train], label='Weighted RAW L1')
    axes[0].legend()
    axes[0].set(xlabel='Optimizer step', ylabel='Training loss', title='B0 / D1 training')
    axes[1].plot([r['step'] for r in train], [r['lr'] for r in train])
    axes[1].set(xlabel='Optimizer step', ylabel='Learning rate')
    fig.tight_layout(); fig.savefig(run / 'training_curves.png', dpi=150); plt.close(fig)
    evaluations = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
    scenes = sorted(evaluations[0]['scene_metrics'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, split in enumerate(('val', 'test')):
        selected = [r for r in evaluations if r['split'] == split]
        for j, metric in enumerate(('psnr', 'ssim')):
            for scene in scenes:
                axes[i, j].plot([r['step'] for r in selected],
                                [r['scene_metrics'][scene][metric] for r in selected], label=scene)
            axes[i, j].set(title=split + ' / ' + metric.upper(), xlabel='Optimizer step')
            axes[i, j].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run / 'scene_metric_curves.png', dpi=150); plt.close(fig)


@torch.no_grad()
def native_artifacts(model, config, run):
    ds = dataset_for_config(config, 'val')
    norm = ds.recipe['raw_normalization']
    chosen = {}
    for row in ds.records:
        chosen.setdefault(row['scene_id'], row)
    output = run / 'native_qualitative'
    output.mkdir(exist_ok=True)
    records, first_input = [], None
    ffmpeg = shutil.which('ffmpeg')
    for scene, row in sorted(chosen.items()):
        # Explicit full-native qualitative protocol, separate from cropped D1 metrics.
        raw_cache = np.load(Path(config['data_root']) / row['raw']['path'], mmap_mode='r', allow_pickle=False)
        frame = raw_cache[row['frame_id']].astype(np.float32)
        raw = torch.from_numpy((frame - norm['offset']) / norm['scale'])[None, None].cuda()
        torch.cuda.synchronize()
        started = time.monotonic()
        prediction = model(raw)
        torch.cuda.synchronize()
        forward_seconds = time.monotonic() - started
        assert prediction.shape == (1, 1, 3072, 3840)
        array = uint8_image(prediction)
        Image.fromarray(array).save(output / (scene + '_sr_x3.png'))
        with Image.open(Path(config['data_root']) / row['target']['path']) as im:
            teacher = im.copy()
        reference = teacher.resize((3840, 3072), Image.Resampling.BICUBIC)
        reference.save(output / (scene + '_teacher_bicubic_reference.png'))
        panels = [reference.copy(), Image.fromarray(array)]
        for im in panels:
            im.thumbnail((960, 768), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1920, 800), '#202020')
        draw = ImageDraw.Draw(canvas)
        canvas.paste(panels[0], (0, 28)); canvas.paste(panels[1], (960, 28))
        draw.text((8, 8), 'Teacher bicubic x3: visual reference, NOT true HR', fill='white')
        draw.text((968, 8), 'B0 native RAW -> x3: qualitative only', fill='white')
        canvas.save(output / (scene + '_overview.jpg'), quality=92)
        # Full-resolution matched crop, no resizing, to inspect edges and artifacts.
        left, top, width, height = 1536, 1536, 768, 768
        detail = Image.new('RGB', (1536, 800), '#202020')
        detail.paste(reference.crop((left, top, left+width, top+height)), (0, 28))
        detail.paste(Image.fromarray(array).crop((left, top, left+width, top+height)), (768, 28))
        ImageDraw.Draw(detail).text((8, 8), 'Reference | B0 native-resolution fixed crop', fill='white')
        detail.save(output / (scene + '_detail.jpg'), quality=95)
        record = {'scene_id': scene, 'raw_path': row['raw']['path'], 'frame_id': row['frame_id'],
                  'output_shape': [3072, 3840], 'single_forward_wall_seconds': forward_seconds,
                  'out_of_range_fraction': float(((prediction < 0) | (prediction > 1)).float().mean()),
                  'metrics': None, 'scope': 'native qualitative; interpolated teacher is not HR truth'}
        if ffmpeg:
            path = output / (scene + '_12frames_preview.mp4')
            command = [ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', '2560x1024',
                       '-r', '6', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                       '-pix_fmt', 'yuv420p', '-y', str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            first_frame = min(len(raw_cache)-12, len(raw_cache)//2)
            try:
                for frame_id in range(first_frame, first_frame+12):
                    native = (raw_cache[frame_id].astype(np.float32)-norm['offset'])/norm['scale']
                    predicted = uint8_image(model(torch.from_numpy(native)[None, None].cuda()))
                    display = np.asarray(Image.fromarray(predicted).resize((1280,1024), Image.Resampling.LANCZOS))
                    target_path = Path(config['data_root']) / row['target']['path']
                    with Image.open(target_path.with_name('%06d.png' % frame_id)) as im:
                        teacher_frame = np.asarray(im)
                    process.stdin.write(np.concatenate((teacher_frame, display), axis=1).tobytes())
            finally:
                process.stdin.close()
            assert process.wait() == 0
            record['preview'] = {'path': path.name, 'frames': 12, 'first_frame': first_frame,
                                 'playback_fps': 6, 'acquisition_fps': None,
                                 'layout': 'teacher original left, native SR reduced for overview right'}
        if first_input is None:
            first_input = raw
        records.append(record)
        print(json.dumps({'event': 'native_visual_complete', 'scene': scene}), flush=True)
    for _ in range(5):
        model(first_input)
    torch.cuda.synchronize()
    timings = []
    for _ in range(20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); model(first_input); end.record(); end.synchronize()
        timings.append(start.elapsed_time(end))
    atomic_json(output / 'manifest.json', {'records': records, 'metric_scope': 'no true native HR metrics',
                'gpu_timing': {'device': config['gpu_uuid'], 'dtype': 'FP32', 'batch_size': 1,
                               'input': [1,1,1024,1280], 'median_ms': float(np.median(timings)),
                               'p95_ms': float(np.percentile(timings,95)), 'samples_ms': timings,
                               'scope': 'model-only CUDA events; excludes copies/postprocess; not SS928'}})
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    config = json.loads((run / 'config.json').read_text())
    index = json.loads((run / 'checkpoint_index.json').read_text())
    selected = index['best']
    if not selected:
        raise RuntimeError('No validation-selected checkpoint')
    path = run / selected['path']
    if sha(path) != selected['sha256']:
        raise RuntimeError('Selected checkpoint hash changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = inference_model(config, state['model']).cuda().eval()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    deployed = to_deploy(model)
    probe = torch.randn(1, 1, 128, 128, device='cuda')
    reference, fused = model(probe), deployed(probe)
    fusion_error = float((reference - fused).abs().max())
    assert fusion_error < 1e-3, fusion_error
    artifacts = run / 'artifacts'; artifacts.mkdir(exist_ok=True)
    deploy_path = artifacts / 'b0_deploy.pt'
    torch.save({'model': {k:v.cpu() for k,v in deployed.state_dict().items()}, 'config': config,
                'parent_checkpoint_sha256': selected['sha256'], 'deploy': True}, deploy_path)
    val = evaluate(model, config['data_root'], 'val', 'cuda', run/'selected_best/val', selected['step'], True,
                   scene_ids=config.get('scene_ids'))
    test = evaluate(model, config['data_root'], 'test', 'cuda', run/'selected_best/test', selected['step'], True,
                    scene_ids=config.get('scene_ids'))
    native = native_artifacts(deployed, config, run)
    export_status = {'status': 'not_exported', 'reason': 'onnx package absent'}
    if importlib.util.find_spec('onnx'):
        import onnx
        deployed.down = ExportGrayUnshuffle()
        destination = artifacts / 'b0_gray_x3_1024x1280.onnx'
        torch.onnx.export(deployed, torch.zeros(1,1,1024,1280,device='cuda'), destination,
                          input_names=['raw'], output_names=['display'], opset_version=17, dynamo=False)
        onnx.checker.check_model(str(destination))
        export_status = {'status': 'exported_checker_passed', 'path': destination.name, 'sha256': sha(destination),
                         'runtime_parity': 'not_run', 'SS928': 'not_verified'}
        if importlib.util.find_spec('onnxruntime'):
            import onnxruntime as ort
            options = ort.SessionOptions(); options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(destination), options, providers=['CPUExecutionProvider'])
            probe = np.random.default_rng(config['seed']).normal(0,1,(1,1,1024,1280)).astype(np.float32)
            expected = deployed(torch.from_numpy(probe).cuda()).cpu().numpy()
            actual = session.run(None, {'raw': probe})[0]
            error = float(np.max(np.abs(actual-expected)))
            export_status['runtime_parity'] = {'max_abs_error': error, 'passed': error < 1e-3}
            assert error < 1e-3, error
    make_curves(run)
    report = {'status': 'completed', 'selected_checkpoint': selected, 'validation': val['scene_metrics'],
              'test': test['scene_metrics'], 'validation_macro': val['macro'], 'test_macro': test['macro'],
              'fusion_max_abs_error_random_128_probe': fusion_error, 'deployment_weight_sha256': sha(deploy_path),
              'native_visual_scenes': len(native), 'onnx': export_status,
              'SS928_latency_status': 'not_measured', 'finished_at_server_utc': utc_now()}
    atomic_json(run / 'result.json', report)
    lines = ['# RT4KSR B0 / D1 训练结果：'+config['model'], '', '验证集选择的检查点：step '+str(selected['step'])+'。',
             '', '| 场景 | 验证PSNR | 验证SSIM | 测试PSNR | 测试SSIM | 范围 |',
             '|---|---:|---:|---:|---:|---|']
    for scene, a in val['scene_metrics'].items():
        b = test['scene_metrics'][scene]
        lines.append('| %s | %.3f | %.5f | %.3f | %.5f | %s |' % (scene,a['psnr'],a['ssim'],b['psnr'],b['ssim'],a['evaluation_scope']))
    lines += ['', '测试为固定间隔开发监控，未用于选择checkpoint。目标为JPEG/增强伪GT，指标不代表物理真实HR恢复。',
              '', '完整RAW推理样例位于native_qualitative；教师双三次插值仅作视觉参考。空间开发子集不代表独立泛化。',
              '', '模型、日志和配方见config.json、checkpoint_index.json、events.jsonl、metrics.jsonl与tensorboard/。',
              '', 'SS928未上板；服务器GPU计时不能作为16.7ms目标达标证明。']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
