"""GPU-0-only training sanity checks; weights are never reused by the main run."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from ir_sr.data import RawDisplayDataset
from ir_sr.model import RT4KSRB0, to_deploy
from ir_sr.auxiliary import training_model, loss_terms, validate_middle_lock, ModuleTimer
from ir_sr.training import (atomic_json, seed_all, validate_data_lock, optimizer_for,
                           save_checkpoint, restore_checkpoint, epoch_loader, dataset_for_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    lock = validate_data_lock(config['data_root'], code)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_all(config['seed'])
    ds = dataset_for_config(config, 'train', use_cache=True)
    middle_lock = validate_middle_lock(config, ds)
    scene_ids = sorted({r['scene_id'] for r in ds.records})
    indices = [next(i for i, r in enumerate(ds.records) if r['scene_id'] == s) for s in scene_ids]
    items = [ds[i] for i in indices]
    raw = torch.stack([r['raw'] for r in items]).cuda()
    gt = torch.stack([r['gt'] for r in items]).cuda()
    middle = torch.stack([r['middle_raw'] for r in items]).cuda() if 'middle_raw' in items[0] else None
    model = training_model(config).cuda()
    optimizer = optimizer_for(model, config)
    losses = []

    def update(m, opt, x, y, z=None):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _, _ = loss_terms(m, x, y, z, config.get('auxiliary_raw_weight', 0))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite preflight loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        return float(loss), float(gradient)

    start = time.monotonic()
    for step in range(150):
        loss, gradient = update(model, optimizer, raw, gt, middle)
        losses.append(loss)
        if (step + 1) % 25 == 0:
            print(json.dumps({'preflight_step': step + 1, 'loss': loss, 'gradient_norm': gradient}), flush=True)
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * .8, 'Tiny-set fit did not improve sufficiently'
    save_checkpoint(args.output / 'resume_probe.pt', model, optimizer, config,
                    {'step': 150, 'epoch': 0, 'next_batch': 0}, -1)
    update(model, optimizer, raw, gt, middle)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    resumed = training_model(config).cuda()
    resumed_optimizer = optimizer_for(resumed, config)
    restore_checkpoint(args.output / 'resume_probe.pt', resumed, resumed_optimizer, config)
    update(resumed, resumed_optimizer, raw, gt, middle)
    resume_max_error = max(float((v.cpu() - expected[k]).abs().max()) for k, v in resumed.state_dict().items())
    assert resume_max_error < 1e-6, resume_max_error
    deployed = to_deploy(resumed)
    with torch.no_grad():
        ref, fused = resumed(raw.float()), deployed(raw.float())
        fusion_max_error = float((ref - fused).abs().max())
        assert fusion_max_error < 1e-4, fusion_max_error
    del model, optimizer, resumed_optimizer, deployed
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    benchmark_model = training_model(config).cuda()
    timer = ModuleTimer(benchmark_model)
    module_samples = []
    benchmark_optimizer = optimizer_for(benchmark_model, config)
    loader, total_batches = epoch_loader(ds, config, 1)
    loader_resume, _ = epoch_loader(ds, config, 1, start_batch=2)
    assert loader.batch_sampler[2:] == loader_resume.batch_sampler
    times = []
    iterator = iter(loader)
    for step in range(min(20, total_batches)):
        begin = time.monotonic()
        batch = next(iterator)
        x, y = batch['raw'].cuda(non_blocking=True), batch['gt'].cuda(non_blocking=True)
        z = batch['middle_raw'].cuda(non_blocking=True) if 'middle_raw' in batch else None
        timer.start_step(step in (0, 5))
        loss, _ = update(benchmark_model, benchmark_optimizer, x, y, z)
        measured = timer.finish()
        if measured is not None:
            module_samples.append({'benchmark_step': step, 'forward_module_ms': measured})
        torch.cuda.synchronize()
        times.append(time.monotonic() - begin)
    del iterator, loader, loader_resume
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    atomic_json(args.output / 'environment.json', {'python': sys.version, 'packages': packages,
                 'gpu_query': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
                                                       '--format=csv,noheader'], text=True)})
    result = {'status': 'passed', 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip(),
              'data_lock_sha256': lock, 'gpu_uuid': config['gpu_uuid'], 'visible_devices': torch.cuda.device_count(),
              'middle_gt_index_sha256': middle_lock, 'geometric_augmentation': ds.augmentation,
              'auxiliary_raw_weight': config.get('auxiliary_raw_weight', 0),
              'module_timing_samples': module_samples,
              'tiny_fit_initial_mean_l1': sum(losses[:10]) / 10, 'tiny_fit_final_mean_l1': sum(losses[-10:]) / 10,
              'checkpoint_resume_max_parameter_error': resume_max_error,
              'fusion_max_abs_error_after_training': fusion_max_error,
              'parameter_count': sum(p.numel() for p in benchmark_model.parameters()),
              'batch_size': config['batch_size'], 'workers': config['workers'], 'benchmark_steps': times,
              'train_crop_hr': ds.train_crop_hr, 'input_shape': list(x.shape), 'target_shape': list(y.shape),
              'steady_step_seconds': sum(times[5:]) / len(times[5:]),
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
              'elapsed_seconds': time.monotonic() - start, 'weights_used_for_main_training': False}
    atomic_json(args.output / 'preflight.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
