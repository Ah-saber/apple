"""Detached process supervisor: alive/resource logs, exit receipt, bounded GPU lease."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import hashlib
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=float, required=True)
    parser.add_argument('--parent-run', type=Path, help='Inherit a completed pilot into a new, independent run')
    parser.add_argument('--lease-fd', type=int, help='GPU lease inherited from the owning serial queue')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == config['gpu_uuid']
    lease_path = args.output.parent / 'TASK-019-gpu0.lock'
    lease = lease_path.open('a') if args.lease_fd is None else os.fdopen(os.dup(args.lease_fd), 'a')
    if args.lease_fd is not None:
        actual, expected = os.fstat(lease.fileno()), lease_path.stat()
        assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-u', '-B', 'tools/train_b0.py', '--config', str(args.config),
               '--output', str(args.output), '--max-wall-seconds', str(args.max_wall_seconds)]
    if args.parent_run:
        parent = args.parent_run.resolve()
        if parent == args.output.resolve():
            raise ValueError('Continuation must use a new run directory')
        completion = json.loads((parent / 'completed.json').read_text())
        receipt = json.loads((parent / 'exit.json').read_text())
        assert completion['status'] == 'completed' and receipt['exit_code'] == 0
        index = json.loads((parent / 'checkpoint_index.json').read_text())
        assert index['last']['step'] == completion['steps'] and index['best'] is not None
        sys.path.insert(0, str(code / 'src'))
        from ir_sr.training import validate_resume_configuration
        previous = json.loads((parent / 'config.json').read_text())
        validate_resume_configuration(previous, config, {'step': index['last']['step']}, True)
        inherited = []
        for entry in index['checkpoints']:
            source = (parent / entry['path']).resolve()
            assert source.is_relative_to(parent / 'checkpoints')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest == entry['sha256']
            destination = args.output / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inherited.append(dict(path=entry['path'], sha256=digest, step=entry['step']))
        # Aliases point to the same immutable best/last snapshots in the new run.
        shutil.copy2(args.output / index['best']['path'], args.output / 'checkpoints/best.pt')
        shutil.copy2(args.output / index['last']['path'], args.output / 'checkpoints/last.pt')
        (args.output / 'checkpoint_index.json').write_text(json.dumps(index, indent=2) + '\n')
        (args.output / 'continuation_parent.json').write_text(json.dumps(dict(
            parent_run=str(parent), inherited=inherited, parent_config=previous,
            parent_training_identity=json.loads((parent / 'training_identity.json').read_text()),
            config_changes={k: dict(before=previous.get(k), after=config.get(k))
                            for k in set(previous) | set(config) if previous.get(k) != config.get(k)},
            note='Model, Adam moments, RNG, epoch and next batch restored; LR horizon unchanged.'
        ), indent=2) + '\n')
        command += ['--resume', str(args.output / index['last']['path']), '--extend-schedule']
    start = time.monotonic()
    hard_timeout = args.max_wall_seconds + 1800
    child_env = dict(os.environ)
    child_env['SR_TRAINING_DEADLINE_MONOTONIC'] = str(start + args.max_wall_seconds)
    with (args.output / 'train.log').open('x') as log:
        process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=child_env)
        (args.output / 'launch.json').write_text(json.dumps({'supervisor_pid': os.getpid(),
                'training_pid': process.pid, 'command': command, 'worktree': str(code),
                'gpu_uuid': config['gpu_uuid'], 'max_training_seconds': args.max_wall_seconds,
                'hard_process_timeout_seconds': hard_timeout}, indent=2) + '\n')
        killed = False
        while process.poll() is None:
            try:
                gpu = subprocess.check_output(['nvidia-smi', '-i', config['gpu_uuid'],
                        '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
            except Exception as error:
                gpu = str(error)
            record = {'elapsed_seconds': time.monotonic() - start, 'process_alive': process.poll() is None,
                      'gpu_memory_mib_util_percent_temp_c_power_w': gpu,
                      'log_bytes': (args.output / 'train.log').stat().st_size}
            with (args.output / 'supervision.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if time.monotonic() - start >= hard_timeout:
                log.flush()
                with (args.output / 'supervision.jsonl').open('a') as stream:
                    stream.write(json.dumps({'event': 'hard_timeout_SIGTERM_own_process_group'}) + '\n')
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                killed = True
                break
            time.sleep(30)
        exit_code = process.wait()
    (args.output / 'exit.json').write_text(json.dumps({'exit_code': exit_code,
            'hard_timeout': killed, 'elapsed_seconds': time.monotonic() - start}, indent=2) + '\n')
    # An inherited descriptor shares the queue's open-file lock. Only its owner unlocks it.
    if args.lease_fd is None:
        fcntl.flock(lease, fcntl.LOCK_UN)
    lease.close()


if __name__ == '__main__':
    main()
