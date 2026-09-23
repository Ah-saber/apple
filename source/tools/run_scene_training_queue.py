"""One frozen, non-restarting GPU-0 queue: preflight, train, test and deliver three models."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ir_sr.training import atomic_json, sha, utc_now


def validate_jobs(plan, code, output):
    jobs = plan['jobs']
    expected = [['day_normal'], ['weather_light', 'weather_medium'],
                ['weather_heavy', 'weather_heavy_c32']]
    if len(jobs) != 3:
        raise ValueError('Exactly the three authorized models are required')
    paths = [output]
    for job, scenes in zip(jobs, expected):
        config = json.loads((code / job['config']).read_text())
        if config['scene_ids'] != scenes or config['gpu_uuid'] != plan['gpu_uuid']:
            raise ValueError('Scene or GPU identity mismatch')
        for key, value in {'seed': 928, 'max_steps': 200000, 'batch_size': 16,
                           'train_crop_hr': [768, 768], 'max_wall_seconds': 21600}.items():
            if config[key] != value:
                raise ValueError('Unapproved training recipe: ' + key)
        paths += [Path(job['run']), Path(job['finish'])]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Output paths must be distinct')
    if any(p.exists() for p in paths):
        raise FileExistsError('Queue cannot overwrite or automatically resume existing outputs')


def require_complete_training(run, max_steps):
    exit_receipt = json.loads((run / 'exit.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    if (exit_receipt['exit_code'] != 0 or exit_receipt['hard_timeout'] or
            completed['status'] != 'completed' or completed['steps'] != max_steps or
            completed['reason'] != 'max_steps'):
        raise RuntimeError('Training failed or stopped short; later models must not launch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip():
        raise RuntimeError('Queue worktree must be clean')
    validate_jobs(plan, code, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    queue_lease = (args.output.parent / 'TASK-021-queue.lock').open('a')
    fcntl.flock(queue_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    begin = time.monotonic()
    current = {'job': None}

    def status(phase, **extra):
        atomic_json(args.output / 'status.json', {'phase': phase, **current,
                    'elapsed_seconds': time.monotonic() - begin, 'server_utc': utc_now(), **extra})

    def command(name, argv, *, gpu=False, timeout=600, lease=None):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if gpu else '')
        record = {'command': argv, 'cwd': str(code), 'gpu': gpu, 'started_at_server_utc': utc_now()}
        atomic_json(args.output / (name + '_command.json'), record)
        started = time.monotonic()
        with (args.output / (name + '.log')).open('x') as log:
            process = subprocess.Popen(argv, cwd=code, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       pass_fds=() if lease is None else (lease.fileno(),))
            atomic_json(args.output / 'active_process.json', dict(record, pid=process.pid))
            # Managed training has its own monotonic deadline and process-group hard timeout.
            # Do not kill that supervisor from here and orphan its training child.
            try:
                result = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
        atomic_json(args.output / (name + '_exit.json'), {'exit_code': result,
                    'elapsed_seconds': time.monotonic() - started})
        if result:
            raise RuntimeError(name + ' failed; queue stopped, see its log')

    @contextmanager
    def gpu_lease():
        lease = (args.output.parent / 'TASK-019-gpu0.lock').open('a')
        deadline = time.monotonic() + plan['gpu_wait_seconds']
        acquired = False
        try:
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    status('waiting_for_gpu_lease')
                    time.sleep(30)
                    continue
                gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
                                               '--format=csv,noheader'], text=True, timeout=10).strip()
                if gpu != plan['gpu_uuid']:
                    raise RuntimeError('Physical GPU 0 UUID changed')
                active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                  '--format=csv,noheader'], text=True, timeout=10)
                if gpu not in active:
                    break
                fcntl.flock(lease, fcntl.LOCK_UN)
                acquired = False
                status('waiting_for_gpu0_idle', active_gpu0=[s for s in active.splitlines() if gpu in s])
                time.sleep(30)
            else:
                raise TimeoutError('GPU 0 remained busy; no foreign process was stopped')
            yield lease
        finally:
            if acquired:
                fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()

    try:
        atomic_json(args.output / 'plan.json', plan)
        atomic_json(args.output / 'launch.json', {'pid': os.getpid(), 'worktree': str(code),
                    'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip(),
                    'plan_sha256': sha(args.plan), 'started_at_server_utc': utc_now(),
                    'gpu_uuid': plan['gpu_uuid'], 'queue_policy': 'serial; stop on failure; no automatic restart'})
        # Verify every subset before spending the formal training budget.
        with gpu_lease():
            for job in plan['jobs']:
                current['job'] = job['name']
                config = code / job['config']
                status('preflight_geometry')
                command(job['name'] + '_geometry', [sys.executable, '-u', '-B',
                        'tools/verify_training_geometry.py', '--config', str(config), '--output',
                        str(args.output / job['name'] / 'geometry')])
                status('preflight_gpu')
                command(job['name'] + '_preflight', [sys.executable, '-u', '-B',
                        'tools/preflight_training.py', '--config', str(config), '--output',
                        str(args.output / job['name'] / 'preflight')], gpu=True)
        completed_jobs = []
        for job in plan['jobs']:
            current['job'] = job['name']
            config_path = code / job['config']
            config = json.loads(config_path.read_text())
            run, finish = Path(job['run']), Path(job['finish'])
            with gpu_lease() as lease:
                status('training', run=str(run), completed_jobs=completed_jobs)
                command(job['name'] + '_training', [sys.executable, '-u', '-B',
                        'tools/run_managed_training.py', '--config', str(config_path), '--output', str(run),
                        '--max-wall-seconds', str(config['max_wall_seconds']), '--lease-fd', str(lease.fileno())],
                        gpu=True, timeout=None, lease=lease)
                require_complete_training(run, config['max_steps'])
                status('finishing', run=str(run), finish=str(finish), completed_jobs=completed_jobs)
                command(job['name'] + '_finish', [sys.executable, '-u', '-B', 'tools/finish_scene_run.py',
                        '--run', str(run), '--baseline', plan['baseline'], '--output', str(finish)],
                        gpu=False, timeout=2400)
                delivery = json.loads((run.parent / (run.name + '-delivery.json')).read_text())
                if sha(delivery['archive']) != delivery['sha256']:
                    raise RuntimeError('Delivery archive hash mismatch')
                completed_jobs.append({'name': job['name'], 'run': str(run), 'finish': str(finish),
                                       'delivery': delivery})
        current['job'] = None
        status('completed', completed_jobs=completed_jobs)
        atomic_json(args.output / 'completed.json', {'status': 'completed', 'jobs': completed_jobs,
                    'elapsed_seconds': time.monotonic() - begin, 'finished_at_server_utc': utc_now()})
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        fcntl.flock(queue_lease, fcntl.LOCK_UN)
        queue_lease.close()


if __name__ == '__main__':
    main()
