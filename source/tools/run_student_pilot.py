"""Authorized shared-GPU sequential pilot, with bounded allocator and no job interruption."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
C=Path(__file__).resolve().parents[1];B=Path('/data/zhangbenzhuang/huawei_sr');A=B/'runs/SS928-STUDENT-V1-20260923-PREFLIGHT'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=C,text=True).strip()
A.mkdir(exist_ok=False);lease=(B/'runs/TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1';env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
python='/data/zhangbenzhuang/miniconda3/envs/restormer/bin/python';start=time.time()
def write(name,value):(A/name).write_text(json.dumps(value,indent=2)+'\n')
def command(tag,cmd,environment=env,timeout=None):
 write('status.json',dict(phase=tag,pid=os.getpid(),started_unix=start,command=cmd))
 with (A/(tag+'.log')).open('x') as f:
  r=subprocess.run(cmd,cwd=C,env=environment,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),timeout=timeout)
 write(tag+'_exit.json',dict(exit=r.returncode,finished_unix=time.time()))
 if r.returncode:raise RuntimeError(tag+' failed')
try:
 free=float(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>5120,free
 write('launch.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),gpu_state=subprocess.check_output(['nvidia-smi'],text=True),allocator_limit_gib=3.5))
 deps='/data/zhangbenzhuang/huawei_sr/runs/TASK-019-export-dependencies/site-packages'
 command('unit_tests',[python,'-B','-m','unittest','discover','-s','tests','-v'],dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(C/'src')+':'+deps),300)
 command('gpu_preflight',[python,'-u','-B','tools/preflight_student.py','--output',str(A/'verification.json')],timeout=600)
 for tag in ['s01','s02']:
  config=C/f'configs/train/student_{tag}_light_medium_8k.json';out=B/f'runs/SS928-STUDENT-V1-20260923-{tag.upper()}-LIGHT_MEDIUM-8K'
  assert not out.exists();free=float(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>5120,free
  command(tag+'_training',[python,'-u','-B','tools/run_managed_training.py','--config',str(config),'--output',str(out),'--max-wall-seconds','3600','--lease-fd',str(lease.fileno())])
  completed=json.loads((out/'completed.json').read_text());assert completed['steps']==8000 and completed['status']=='completed'
 write('status.json',dict(phase='completed',seconds=time.time()-start))
except BaseException:
 write('status.json',dict(phase='failed',error=traceback.format_exc(),seconds=time.time()-start));raise
