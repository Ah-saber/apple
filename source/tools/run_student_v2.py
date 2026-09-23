"""Sequential equal-budget training after complete architecture/ONNX/GPU screening."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
C=Path(__file__).resolve().parents[1];R=Path('/data/zhangbenzhuang/huawei_sr/runs');A=R/'SS928-STUDENT-V2-20260923-PREFLIGHT/queue_s06';P='/data/zhangbenzhuang/miniconda3/envs/restormer/bin/python'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=C,text=True).strip()
screen=json.loads((A.parent/'screen_s06/report.json').read_text());assert screen['status']=='passed'
for tag in ['s06']:
 assert screen['models'][tag]['mean_ms'] < screen['models']['s02']['mean_ms']*.95
 assert len(screen['models'][tag]['training_probe_losses'])==3
A.mkdir(exist_ok=False);lease=(R/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1';env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');started=time.monotonic()
def write(name,obj):(A/name).write_text(json.dumps(obj,indent=2))
write('launch.json',dict(pid=os.getpid(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),gpu_before=subprocess.check_output(['nvidia-smi'],text=True)))
try:
 for tag in ['s06']:
  free=float(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>5120,free
  out=R/f'SS928-STUDENT-V2-20260923-{tag.upper()}-LIGHT_MEDIUM-8K';assert not out.exists();config=C/f'configs/train/student_{tag}_light_medium_8k.json'
  write('status.json',dict(phase=tag+'_training',seconds=time.monotonic()-started))
  cmd=[P,'-u','-B','tools/run_managed_training.py','--config',str(config),'--output',str(out),'--max-wall-seconds','3600','--lease-fd',str(lease.fileno())]
  with (A/(tag+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
  complete=json.loads((out/'completed.json').read_text());assert r.returncode==0 and complete['steps']==8000 and complete['status']=='completed'
 write('status.json',dict(phase='completed',seconds=time.monotonic()-started))
except BaseException:write('status.json',dict(phase='failed',error=traceback.format_exc()));raise
