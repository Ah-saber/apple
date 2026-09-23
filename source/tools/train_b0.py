"""Time-bounded, scene-monitored B0 training from a frozen worktree."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ir_sr.data import RawDisplayDataset
from ir_sr.model import RT4KSRB0
from ir_sr.auxiliary import training_model, loss_terms, validate_middle_lock, ModuleTimer
from ir_sr.training import (atomic_json, append_json, seed_all, validate_data_lock, optimizer_for,
                           epoch_loader, learning_rate, save_checkpoint, restore_checkpoint,
                           evaluate, sha, utc_now, dataset_for_config, record_selected_data)


def train(args):
    start = time.monotonic()
    deadline = float(os.environ.get('SR_TRAINING_DEADLINE_MONOTONIC', start + args.max_wall_seconds))
    config = json.loads(args.config.read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    code = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip():
        raise RuntimeError('Training worktree must be clean')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    lock_sha = validate_data_lock(config['data_root'], code)
    target_cache_sha = None
    if config.get('target_cache_root'):
        cache_root = Path(config['target_cache_root'])
        cache_index = json.loads((cache_root / 'cache_index.json').read_text())
        assert cache_index['data_lock_sha256'] == lock_sha and cache_index['status'] == 'complete'
        for entry in cache_index['sequences']:
            assert sha(cache_root / entry['path']) == entry['sha256'], 'Target cache changed'
        target_cache_sha = sha(cache_root / 'cache_index.json')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists() and not args.resume:
        raise RuntimeError('Existing training run requires explicit resume')
    atomic_json(args.output / 'config.json', config)
    seed_all(config['seed'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    if config.get('cuda_memory_limit_gib'):
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_limit_gib'] * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model = training_model(config).cuda()
    module_timer = ModuleTimer(model)
    optimizer = optimizer_for(model, config)
    progress = {'step': 0, 'epoch': 0, 'next_batch': 0}
    best_score = -1.0
    if args.resume:
        progress, best_score = restore_checkpoint(args.resume, model, optimizer, config, args.extend_schedule)
    dataset = dataset_for_config(config, 'train', use_cache=True)
    middle_lock_sha = validate_middle_lock(config, dataset)
    if dataset.sequence_normalization:
        dataset.sequence_normalization.validate_sources(config['data_root'])
    selection_sha = record_selected_data(config, args.output)
    writer = SummaryWriter(str(args.output / 'tensorboard'), flush_secs=20)
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append('SIGINT'))
    atomic_json(args.output / 'training_identity.json', {
        'code_commit': commit, 'worktree': str(code), 'data_lock_sha256': lock_sha,
        'target_cache_index_sha256': target_cache_sha,
        'middle_gt_index_sha256': middle_lock_sha,
        'sequence_normalization_sha256': config.get('sequence_normalization_sha256'),
        'augmentation': config.get('geometric_augmentation', False),
        'auxiliary_raw_weight': config.get('auxiliary_raw_weight', 0),
        'selected_data_sha256': selection_sha, 'scene_ids': config.get('scene_ids'),
        'train_crop_hr': dataset.train_crop_hr,
        'train_crop_lr': [d // config['scale'] for d in dataset.train_crop_hr],
        'started_at_server_utc': utc_now(), 'gpu_uuid': config['gpu_uuid'],
        'torch': torch.__version__, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'max_wall_seconds': args.max_wall_seconds, 'clock_basis': 'monotonic duration, host-calculated budget',
        'initialization': 'random' if not args.resume else str(args.resume)})
    index = {'checkpoints': [], 'best': None, 'last': None}
    if args.resume and (args.output / 'checkpoint_index.json').exists():
        index = json.loads((args.output / 'checkpoint_index.json').read_text())
        if progress['step'] != index['last']['step']:
            raise RuntimeError('Resuming an older checkpoint requires a new run directory')
    loss_window, gradient_window, duration_window, data_window, scene_window = [], [], [], [], Counter()
    display_window, auxiliary_window, augmentation_window = [], [], Counter()
    last_val, last_test = None, None

    def event(kind, **fields):
        record = {'kind': kind, 'server_utc': utc_now(), 'step': progress['step'],
                  'elapsed_seconds': time.monotonic() - start, **fields}
        append_json(args.output / 'events.jsonl', record)
        print(json.dumps(record, allow_nan=False), flush=True)
        return record

    def status(phase, **fields):
        atomic_json(args.output / 'status.json', {
            'phase': phase, 'step': progress['step'], 'epoch': progress['epoch'],
            'next_batch': progress['next_batch'], 'best_val_macro_psnr': best_score,
            'server_utc': utc_now(), 'elapsed_seconds': time.monotonic() - start,
            'remaining_training_seconds': max(0, deadline - time.monotonic()), **fields})

    def checkpoint(best=False):
        checkpoint_started = time.monotonic()
        path = args.output / 'checkpoints' / ('step_%09d.pt' % progress['step'])
        if path.exists():
            digest = sha(path)
            prior = next(r for r in index['checkpoints'] if r['step'] == progress['step'])
            if prior['sha256'] != digest:
                raise RuntimeError('Existing immutable checkpoint changed')
        else:
            digest = save_checkpoint(path, model, optimizer, config, dict(progress), best_score)
        entry = {'step': progress['step'], 'path': str(path.relative_to(args.output)), 'sha256': digest}
        index['checkpoints'] = [r for r in index['checkpoints'] if r['step'] != progress['step']] + [entry]
        index['last'] = entry
        save_checkpoint(args.output / 'checkpoints/last.pt', model, optimizer, config, dict(progress), best_score)
        if best:
            save_checkpoint(args.output / 'checkpoints/best.pt', model, optimizer, config, dict(progress), best_score)
            index['best'] = {**entry, 'val_macro_psnr': best_score}
        atomic_json(args.output / 'checkpoint_index.json', index)
        event('checkpoint_saved', checkpoint_seconds=time.monotonic()-checkpoint_started, best=best)

    def evaluation(split, final=False):
        nonlocal best_score, last_val, last_test
        status('evaluating_' + split)
        folder = args.output / 'evaluation' / ('step_%09d_%s' % (progress['step'], split))
        result = evaluate(model, config['data_root'], split, 'cuda', folder, progress['step'],
                          save_examples=(split == 'val' or final), scene_ids=config.get('scene_ids'), config=config)
        for scene, values in result['scene_metrics'].items():
            for metric in ('psnr', 'ssim', 'output_out_of_range_fraction'):
                writer.add_scalar(split + '/' + scene + '/' + metric, values[metric], progress['step'])
        for metric, value in result['macro'].items():
            writer.add_scalar(split + '/macro/' + metric, value, progress['step'])
        summary = {k: result[k] for k in ('step', 'split', 'scene_metrics', 'macro', 'by_scope', 'elapsed_seconds')}
        append_json(args.output / 'metrics.jsonl', summary)
        event('evaluation', split=split, macro=result['macro'], scene_metrics=result['scene_metrics'],
              evaluation_seconds=result['elapsed_seconds'])
        if split == 'val':
            last_val = result
            if result['macro']['psnr'] > best_score:
                best_score = result['macro']['psnr']
                checkpoint(best=True)
                event('best_checkpoint', selected_by='validation_macro_psnr', score=best_score)
        else:
            last_test = result
        writer.flush()
        status('training')
        return result

    event('training_started', config=config, resume=bool(args.resume))
    status('training')
    reason = 'max_steps'
    try:
        while progress['step'] < config['max_steps']:
            epoch, offset = progress['epoch'], progress['next_batch']
            loader, batches = epoch_loader(dataset, config, epoch, offset)
            iterator = iter(loader)
            for batch_index in range(offset, batches):
                if stop_requested or time.monotonic() >= deadline:
                    reason = stop_requested[-1] if stop_requested else 'training_time_budget'
                    break
                before = time.monotonic()
                batch = next(iterator)
                data_seconds = time.monotonic() - before
                step = progress['step'] + 1
                module_timer.start_step(step == 1 or step % config['log_every'] == 0)
                module_timer.begin('host_to_device')
                raw, target = batch['raw'].cuda(non_blocking=True), batch['gt'].cuda(non_blocking=True)
                middle = batch['middle_raw'].cuda(non_blocking=True) if 'middle_raw' in batch else None
                context = batch['context'].cuda(non_blocking=True) if 'context' in batch else None
                context_box = batch['context_box'].cuda(non_blocking=True) if 'context_box' in batch else None
                raw_loss_multiplier = batch['raw_loss_multiplier'].cuda(non_blocking=True) if config.get('sequence_normalization_index') else None
                module_timer.end('host_to_device')
                lr = learning_rate(step, config)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                module_timer.begin('forward_and_loss')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, display_loss, auxiliary_loss = loss_terms(model, raw, target, middle,
                                                                  config.get('auxiliary_raw_weight', 0), raw_loss_multiplier, context, context_box)
                module_timer.end('forward_and_loss')
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss at step ' + str(step))
                module_timer.begin('backward')
                loss.backward()
                module_timer.end('backward')
                module_timer.begin('gradient_clip_and_optimizer')
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'],
                                                          error_if_nonfinite=True)
                optimizer.step()
                module_timer.end('gradient_clip_and_optimizer')
                timing_sample = module_timer.finish()
                if timing_sample is not None:
                    append_json(args.output / 'module_timing.jsonl', {'step': step, 'cuda_ms': timing_sample,
                                'data_loading_seconds': data_seconds,
                                'scope': 'sampled CUDA event intervals; forward_and_loss contains child module times'})
                loss_value, gradient_value = float(loss.detach()), float(gradient.detach())
                display_window.append(float(display_loss.detach()))
                auxiliary_window.append(float(auxiliary_loss.detach()))
                augmentation_window.update(map(int, batch['augmentation_id']))
                progress.update(step=step, epoch=epoch if batch_index + 1 < batches else epoch + 1,
                                next_batch=batch_index + 1 if batch_index + 1 < batches else 0)
                loss_window.append(loss_value)
                gradient_window.append(gradient_value)
                duration_window.append(time.monotonic() - before)
                data_window.append(data_seconds)
                scene_window.update(batch['scene_id'])
                if step % config['log_every'] == 0 or step == 1:
                    record = event('train_log', loss_l1=float(np.mean(loss_window)),
                                   display_loss_l1=float(np.mean(display_window)),
                                   auxiliary_raw_loss_l1=float(np.mean(auxiliary_window)),
                                   auxiliary_raw_weight=config.get('auxiliary_raw_weight', 0),
                                   augmentation_counts=dict(augmentation_window),
                                   gradient_norm=float(np.mean(gradient_window)), lr=lr,
                                   seconds_per_step=float(np.mean(duration_window)),
                                   data_seconds_per_step=float(np.mean(data_window)),
                                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                                   scene_samples=dict(scene_window))
                    for key in ('loss_l1', 'display_loss_l1', 'auxiliary_raw_loss_l1', 'gradient_norm', 'lr', 'seconds_per_step', 'peak_allocated_gib'):
                        writer.add_scalar('train/' + key, record[key], step)
                    loss_window.clear(); gradient_window.clear(); duration_window.clear(); data_window.clear(); scene_window.clear()
                    display_window.clear(); auxiliary_window.clear(); augmentation_window.clear()
                    status('training', last_loss=loss_value)
                if step % config['validation_every'] == 0 or step in config['early_validation_steps']:
                    evaluation('val')
                if step % config['test_every'] == 0 or step in config['early_test_steps']:
                    evaluation('test')
                if step % config['checkpoint_every'] == 0:
                    checkpoint()
                if step >= config['max_steps']:
                    break
            del iterator, loader
            if stop_requested or time.monotonic() >= deadline or progress['step'] >= config['max_steps']:
                if time.monotonic() >= deadline:
                    reason = 'training_time_budget'
                break
        if last_val is None or last_val['step'] != progress['step']:
            evaluation('val', final=True)
        if last_test is None or last_test['step'] != progress['step']:
            evaluation('test', final=True)
        checkpoint()
        event('training_finished', reason=reason, best=index['best'], last=index['last'])
        status('finalizing', stop_reason=reason)
        writer.close()
        # Same frozen worktree; trained best weights only, no further optimizer steps.
        command = [sys.executable, '-u', '-B', str(code / 'tools/finalize_training.py'), '--run', str(args.output)]
        if config.get('pilot_skip_deployment', False):
            event('pilot_deployment_skipped', reason='Calibration-aware pilot; deployment contract requires explicit sequence metadata')
        else:
            if config.get('sequence_normalization_index'):
                raise RuntimeError('Use a calibration-aware finalizer before deployment')
            subprocess.run(command, cwd=code, check=True)
        status('completed', stop_reason=reason)
        atomic_json(args.output / 'completed.json', {'status': 'completed', 'steps': progress['step'],
                    'reason': reason, 'best': index['best'], 'last': index['last'], 'finished_at_server_utc': utc_now()})
    except Exception:
        writer.close()
        status('failed', error=traceback.format_exc())
        event('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=float, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--extend-schedule', action='store_true', help='Resume a completed prefix without changing the frozen LR curve')
    train(parser.parse_args())
