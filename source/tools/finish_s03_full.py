"""Immutable post-training pipeline; waits for exit receipt, then tests and exports both selections."""
import argparse,hashlib,io,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();A=a.output;A.mkdir(parents=True,exist_ok=False)
W=Path(__file__).resolve().parents[1];R=a.run.parent;P=sys.executable
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();C=A/'code';C.mkdir()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=W))) as archive:archive.extractall(C)
(A/'analysis_identity.json').write_text(json.dumps(dict(code_commit=commit,pid=os.getpid(),code=str(C),run=str(a.run)),indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'TASK-019-export-dependencies/site-packages'))
def status(phase,**kwargs):(A/'status.json').write_text(json.dumps(dict(phase=phase,**kwargs),indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(name,arguments,gpu):
 status(name);started=time.monotonic();cmd=[P,'-u','-B',*arguments]
 (A/(name+'_command.json')).write_text(json.dumps(dict(command=cmd,cwd=str(C),gpu=gpu),indent=2))
 with (A/(name+'.log')).open('x') as f:proc=subprocess.run(cmd,cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'] if gpu else ''),stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 (A/(name+'_exit.json')).write_text(json.dumps(dict(exit_code=proc.returncode,seconds=time.monotonic()-started)))
 if proc.returncode:raise RuntimeError(name+' failed; inspect '+str(A/(name+'.log')))
def entry(key,label,run,record):
 ckpt=run/record['path'];assert sha(ckpt)==record['sha256'];return dict(key=key,label=label,run=str(run),checkpoint=str(ckpt),step=record['step'],sha256=record['sha256'])
try:
 status('waiting_for_training_exit',target_steps=200000)
 deadline=time.monotonic()+23400
 while not (a.run/'exit.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('No training exit receipt within 6.5 hours')
  time.sleep(15)
 receipt=json.loads((a.run/'exit.json').read_text());assert receipt['exit_code']==0,receipt
 completed=json.loads((a.run/'completed.json').read_text());assert completed['status']=='completed' and completed['steps']==200000 and completed['reason']=='max_steps',completed
 index=json.loads((a.run/'checkpoint_index.json').read_text());assert index['best'] and index['last']['step']==200000
 entries=[]
 for key,label,folder in [('baseline','Original reference 8k','GLOBAL-REFERENCE-V1-20260923-LIGHT_MEDIUM-8K'),('s02','S02 8k','SS928-STUDENT-V1-20260923-S02-LIGHT_MEDIUM-8K'),('s03_8k','S03 8k','SS928-STUDENT-V2-20260923-S03-LIGHT_MEDIUM-8K')]:
  runpath=R/folder;idx=json.loads((runpath/'checkpoint_index.json').read_text());e=next(x for x in idx['checkpoints'] if x['step']==8000);entries.append(entry(key,label,runpath,e))
 for key,which in [('s03_best','best'),('s03_last','last')]:entries.append(entry(key,f"S03 {which} {index[which]['step']}",a.run,index[which]))
 models=A/'models.json';models.write_text(json.dumps(dict(models=entries,selection='validation crop macro PSNR only; compare selected best and final200k with frozen8k references'),indent=2))
 run('quality',['tools/evaluate_students.py','--models',str(models),'--output',str(A/'quality')],True)
 timingmodels=A/'timing_models.json';timingmodels.write_text(json.dumps(dict(models=[e for e in entries if e['key'].startswith('s03_')]),indent=2))
 run('timing',['tools/benchmark_students.py','--models',str(timingmodels),'--output',str(A/'timing')],True)
 run('videos',['tools/evaluate_full_videos.py','--models',str(models),'--output',str(A/'videos')],True)
 for which in ('best','last'):
  run('onnx_'+which,['tools/export_student.py','--run',str(a.run),'--checkpoint',str(a.run/index[which]['path']),'--selection-label','validation selected best' if which=='best' else 'last200000','--output',str(A/('onnx_'+which)),'--hierarchical-pooling'],False)
 run('report',['tools/report_full_training.py','--run',str(a.run),'--output',str(A)],False)
 for e in index['checkpoints']:assert sha(a.run/e['path'])==e['sha256']
 (A/'verification.json').write_text(json.dumps(dict(status='passed',training_exit=receipt,completed=completed,checkpoints_verified=len(index['checkpoints']),visual_review='human review pending; media decoding alone does not establish quality'),indent=2))
 link=R/'AUX-RAW-METRICS-20260923/s03_full_training'
 if not link.exists() and not link.is_symlink():link.symlink_to(A,target_is_directory=True)
 status('completed',selected_best_step=index['best']['step'],last_step=200000)
 files=[dict(path=str(f.relative_to(A)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(A.rglob('*')) if f.is_file()]
 (A/'artifact_manifest.json').write_text(json.dumps(dict(code_commit=commit,files=files),indent=2))
except BaseException:status('failed',error=traceback.format_exc());raise
