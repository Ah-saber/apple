"""Four-model serial queue; validate every model before formal training; stop on failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ir_sr.training import atomic_json, sha, utc_now
from run_scene_training_queue import require_complete_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();code=Path(__file__).resolve().parents[1];plan=json.loads(args.plan.read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip():
        raise RuntimeError('Formal queue requires a clean committed version')
    expected=[['day_normal'],['weather_light','weather_medium'],['weather_heavy','weather_heavy_c32'],['night_ordinary','night_special']]
    assert len(plan['jobs'])==4
    paths=[args.output]
    for job,scenes in zip(plan['jobs'],expected):
        c=json.loads((code/job['config']).read_text());assert c['scene_ids']==scenes and c['gpu_uuid']==plan['gpu_uuid']
        assert c['max_steps']==200000 and c['train_crop_hr']==[768,768] and c['batch_size']==16
        assert c['geometric_augmentation'] is True
        assert c['auxiliary_raw_weight']==(0 if scenes==['day_normal'] else .1)
        paths.append(Path(job['run']))
    assert len({x.resolve() for x in paths})==5 and not any(x.exists() for x in paths)
    args.output.mkdir(parents=True)
    lease=(args.output.parent/'TASK-019-gpu0.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert plan['gpu_uuid'] not in active,'GPU0 is occupied; queue does not interrupt other processes'
    actual=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual==plan['gpu_uuid']
    start=time.monotonic(); completed=[];current=None

    def status(phase,**extra):
        atomic_json(args.output/'status.json',{'phase':phase,'current_job':current,'completed_jobs':completed,
                    'elapsed_seconds':time.monotonic()-start,'server_utc':utc_now(),**extra})

    def command(name,argv,gpu=False,timeout=None,inherit=False,export=False):
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if gpu else '',PYTHONDONTWRITEBYTECODE='1')
        if export:
            dependencies=Path('/data/zhangbenzhuang/huawei_sr/runs/TASK-019-export-dependencies/site-packages')
            assert dependencies.is_dir()
            environment['PYTHONPATH']=str(dependencies)
        begin=time.monotonic()
        with (args.output/(name+'.log')).open('x') as log:
            process=subprocess.Popen(argv,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                     pass_fds=(lease.fileno(),) if inherit else ())
            atomic_json(args.output/'active_process.json',{'name':name,'pid':process.pid,'command':argv})
            try:
                result=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        atomic_json(args.output/(name+'_exit.json'),{'exit_code':result,'seconds':time.monotonic()-begin})
        if result:raise RuntimeError(name+' failed; later training is not launched')

    try:
        atomic_json(args.output/'plan.json',plan)
        atomic_json(args.output/'launch.json',{'pid':os.getpid(),'worktree':str(code),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
                    'plan_sha256':sha(args.plan),'gpu_uuid':plan['gpu_uuid'],'started_at_server_utc':utc_now(),
                    'policy':'one GPU, one training at a time, all preflights first, stop on failure'})
        status('unit_tests')
        command('unit_tests',[sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],timeout=300,export=True)
        for job in plan['jobs']:
            current=job['name'];cfg=str(code/job['config'])
            status('verify_augmentation_and_targets')
            command(current+'_data',[sys.executable,'-B','tools/verify_augmented_data.py','--config',cfg,'--output',str(args.output/current/'data')],timeout=900)
            status('gpu_preflight')
            command(current+'_gpu',[sys.executable,'-u','-B','tools/preflight_training.py','--config',cfg,'--output',str(args.output/current/'gpu')],gpu=True,timeout=1200)
        for job in plan['jobs']:
            current=job['name'];cfg=code/job['config'];c=json.loads(cfg.read_text());run=Path(job['run'])
            status('training',run=str(run))
            command(current+'_training',[sys.executable,'-u','-B','tools/run_managed_training.py','--config',str(cfg),
                    '--output',str(run),'--max-wall-seconds',str(c['max_wall_seconds']),'--lease-fd',str(lease.fileno())],gpu=True,inherit=True)
            require_complete_training(run,c['max_steps'])
            result=json.loads((run/'result.json').read_text());assert result['status']=='completed'
            status('exporting',run=str(run))
            command(current+'_onnx',[sys.executable,'-u','-B','tools/export_b0_onnx.py','--run',str(run),
                    '--output',str(run/'onnx')],timeout=1800,export=True)
            exported=json.loads((run/'onnx/export_verification.json').read_text());assert exported['status']=='passed'
            completed.append({'name':current,'run':str(run),'steps':c['max_steps'],'onnx_sha256':exported['onnx_sha256']})
        current=None;status('completed')
        atomic_json(args.output/'completed.json',{'status':'completed','jobs':completed,'seconds':time.monotonic()-start})
    except BaseException:
        status('failed',error=traceback.format_exc());raise
    finally:
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
