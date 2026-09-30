"""Reproducible B0 training helpers, explicit scene evaluation and checkpoint state."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from ir_sr.data import RawDisplayDataset
from ir_sr.metrics import image_metrics


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_data_lock(root, code):
    root, code = Path(root), Path(code)
    lock = json.loads((code / 'manifests/dataset_d1.lock.json').read_text())
    for entry in lock['manifests']:
        if sha(root / entry['path']) != entry['sha256']:
            raise RuntimeError('Frozen data metadata changed: ' + entry['path'])
    for relative, key in [('manifests/dataset_d1/DATASET_READY.json', 'release_record_sha256'),
                          ('qa/dataset_d1/verification.json', 'verification_sha256')]:
        if sha(root / relative) != lock[key]:
            raise RuntimeError('Frozen data verification changed')
    return sha(code / 'manifests/dataset_d1.lock.json')


def epoch_loader(dataset, config, epoch, start_batch=0):
    dataset.set_epoch(epoch)
    indices = list(dataset.balanced_sampler(config['seed'] + epoch))
    size = config['batch_size']
    batches = [indices[i:i+size] for i in range(0, len(indices) - size + 1, size)]
    loader = DataLoader(dataset, batch_sampler=batches[start_batch:], num_workers=config['workers'],
                        persistent_workers=False, pin_memory=True, timeout=120 if config['workers'] else 0,
                        generator=torch.Generator().manual_seed(config['seed'] + 100000 + epoch))
    return loader, len(batches)


def dataset_for_config(config, split, use_cache=False):
    dataset = RawDisplayDataset(config['data_root'], split, dataset_version='dataset_d1',
                               seed=config['seed'],
                               scene_ids=config.get('scene_ids'),
                               train_crop_hr=config.get('train_crop_hr') if split == 'train' else None,
                               geometric_augmentation=config.get('geometric_augmentation', False) if split == 'train' else False,
                               middle_gt_root=config.get('middle_gt_root') if split == 'train' else None,
                               target_cache_root=config.get('target_cache_root') if use_cache else None,
                               sequence_normalization_index=config.get('sequence_normalization_index'),
                               sequence_normalization_sha256=config.get('sequence_normalization_sha256'),
                               global_reference=config.get('global_reference', False),
                               native_training=bool(config.get('native_training_mode')) and split=='train',
                               native_crop_sampling=config.get('native_crop_sampling','center'),
                               absolute_raw_reference=config.get('absolute_raw_reference',False),
                               reference_tone_supervision=config.get('reference_tone_weight',0)>0 and split=='train',
                               reference_sensor_y=config.get('reference_sensor_y',False))
    expected = config.get('expected_split_counts', {}).get(split)
    if expected is not None and dict(Counter(r['scene_id'] for r in dataset.records)) != expected:
        raise ValueError('Selected split counts differ from frozen configuration: ' + split)
    if config.get('input_frames',1)>1:
        from ir_sr.temporal_stack import TemporalStackDataset
        dataset=TemporalStackDataset(dataset,config['input_frames'],split,config.get('causal_dynamic_correction',False))
    return dataset


def record_selected_data(config, output):
    selection = {'dataset': config['dataset'], 'scene_ids': config.get('scene_ids'), 'splits': {}}
    for split in ('train', 'val', 'test'):
        dataset = dataset_for_config(config, split)
        selection['splits'][split] = {'counts': dict(Counter(r['scene_id'] for r in dataset.records)),
                                      'records': dataset.records}
    path = Path(output) / 'selected_data.json'
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError('Saved run subset changed')
    else:
        atomic_json(path, selection)
    return sha(path)


def learning_rate(step, config):
    warmup = config['warmup_steps']
    if step <= warmup:
        return config['lr'] * step / warmup
    fraction = min(1.0, (step - warmup) / max(1, config.get('lr_schedule_steps', config['max_steps']) - warmup))
    return config['min_lr'] + .5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * fraction))


def optimizer_for(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(.9, .99), weight_decay=0)


def save_checkpoint(path, model, optimizer, config, progress, best_score):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
             'progress': progress, 'best_val_macro_psnr': best_score, 'server_utc': utc_now(),
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()},
             'resume_protocol': 'deterministic per-epoch weighted index plan and per-index crop; skip consumed batches'}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    return sha(path)


def validate_resume_configuration(previous, config, progress, extend_schedule=False):
    if previous == config:
        return
    if not extend_schedule:
        raise ValueError('Resume requires identical frozen configuration')
    allowed = {'max_steps', 'max_wall_seconds', 'test_every', 'version'}
    changed = {k for k in set(previous) | set(config) if previous.get(k) != config.get(k)}
    if changed - allowed:
        raise ValueError('Continuation changed training semantics: ' + str(sorted(changed - allowed)))
    horizon = previous.get('lr_schedule_steps')
    if not horizon or config.get('lr_schedule_steps') != horizon:
        raise ValueError('Continuation requires the same explicit LR horizon')
    if not (progress['step'] == previous['max_steps'] < config['max_steps'] <= horizon):
        raise ValueError('Extend only a completed prefix within the frozen LR horizon')
    if config['test_every'] != config['max_steps'] or config['max_wall_seconds'] <= 0:
        raise ValueError('Full-run test must be end-only, with a positive wall-time bound')


def restore_checkpoint(path, model, optimizer, config, extend_schedule=False, native_adaptation=False):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if native_adaptation:
        from ir_sr.native_training import validate_native_adaptation
        validate_native_adaptation(state['config'],config,state['progress'])
    else:
        validate_resume_configuration(state['config'], config, state['progress'], extend_schedule)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch_cpu'])
    torch.cuda.set_rng_state_all(state['rng']['torch_cuda'])
    return state['progress'], state['best_val_macro_psnr']


def uint8_image(tensor):
    return np.rint(tensor.detach().float().cpu().numpy().squeeze().clip(0, 1) * 255).astype(np.uint8)


def save_comparison(path, raw, target, prediction, sequence_normalized=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = raw[:, -1:]
    raw_display = (raw.float() if sequence_normalized else (raw.float() + 3) / 6).clamp(0, 1)
    raw_display = torch.nn.functional.interpolate(raw_display, size=target.shape[-2:], mode='bicubic',
                                                   align_corners=False).clamp(0, 1)
    images = [uint8_image(t) for t in (raw_display, target, prediction.clamp(0, 1),
                                      (prediction.clamp(0, 1) - target).abs() * 4)]
    labels = ['RAW segment normalized' if sequence_normalized else 'RAW fixed DN display', 'Teacher / target', 'B0 prediction', 'Absolute error x4']
    panels = []
    for array in images:
        im = Image.fromarray(array)
        im.thumbnail((480, 480), Image.Resampling.LANCZOS)
        panels.append(im)
    w, h = panels[0].size
    canvas = Image.new('RGB', (4 * (w + 8), h + 34), '#202020')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(panels, labels)):
        canvas.paste(im, (i * (w+8), 28))
        draw.text((i * (w+8) + 2, 8), label, fill='white')
    canvas.save(path)
    Image.fromarray(images[2]).save(path.with_name(path.stem + '_prediction.png'))


@torch.no_grad()
def evaluate(model, root, split, device, output, step, save_examples=False, scene_ids=None, config=None):
    was_training = model.training
    model.eval()
    dataset = dataset_for_config(config, split) if config is not None else RawDisplayDataset(root, split, dataset_version='dataset_d1', scene_ids=scene_ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_image, by_scene, seen = [], defaultdict(list), set()
    start = time.monotonic()
    for index in range(len(dataset)):
        item = dataset[index]
        raw = item['raw'][None].to(device)
        target = item['gt'][None].to(device)
        context_args = {k: item[k][None].to(device) for k in ('context', 'context_box')} if 'context' in item else {}
        prediction = model(raw, **context_args)
        result = image_metrics(prediction, target, border=3)
        row = {'sample_id': item['sample_id'], 'source_sample_id': item['source_sample_id'],
               'scene_id': item['scene_id'], 'evaluation_scope': item['evaluation_scope'],
               'crop_tlhw': item['crop_tlhw'], **result}
        per_image.append(row)
        by_scene[row['scene_id']].append(row)
        if save_examples and row['scene_id'] not in seen:
            save_comparison(output / 'examples' / (row['scene_id'] + '.jpg'), raw, target, prediction, sequence_normalized=bool(dataset.sequence_normalization))
            seen.add(row['scene_id'])
    scenes = {}
    for scene, rows in sorted(by_scene.items()):
        scenes[scene] = {'count': len(rows), 'evaluation_scope': rows[0]['evaluation_scope'],
                         **{key: float(np.mean([r[key] for r in rows])) for key in
                            ('psnr', 'ssim', 'mse', 'output_out_of_range_fraction')}}
    expected = set(scene_ids) if scene_ids is not None else {'day_normal', 'night_ordinary', 'night_special',
                'weather_light', 'weather_medium', 'weather_heavy', 'weather_heavy_c32'}
    assert set(scenes) == expected and all(r['count'] == 12 for r in scenes.values())
    macro = {key: float(np.mean([r[key] for r in scenes.values()])) for key in ('psnr', 'ssim')}
    scopes = {}
    for scope in ('capture_group_development', 'spatial_development'):
        selected = [r for r in scenes.values() if r['evaluation_scope'] == scope]
        if not selected:
            continue
        scopes[scope] = {key: float(np.mean([r[key] for r in selected])) for key in ('psnr', 'ssim')}
    result = {'step': step, 'split': split, 'server_utc': utc_now(), 'scene_metrics': scenes,
              'macro': macro, 'by_scope': scopes, 'images': per_image,
              'elapsed_seconds': time.monotonic() - start,
              'metric_protocol': 'grayscale float [0,1]; predictions clipped, no uint8 rounding; HR border=3; '
                                 'SSIM Gaussian 11 sigma1.5 population covariance, data_range=1; image then scene mean',
              'development_monitoring_not_blind_test': True,
              'sequence_normalization_sha256': config.get('sequence_normalization_sha256') if config else None}
    atomic_json(output / 'metrics.json', result)
    model.train(was_training)
    return result
