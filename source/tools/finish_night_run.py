"""Bounded post-training export/comparison and portable archive for a night run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ir_sr.training import atomic_json, sha, validate_data_lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    code=Path(__file__).resolve().parents[1]
    identity={'pid':os.getpid(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
              'run':str(args.run),'baseline':str(args.baseline),'output':str(args.output),
              'started_at_server_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    atomic_json(args.output/'launch.json',identity)
    # The caller records the new run's full budget, including any larger-patch schedule.
    parser_config = args.run / 'config.json'
    wait_seconds = 9000
    if parser_config.exists():
        wait_seconds = max(wait_seconds, json.loads(parser_config.read_text())['max_wall_seconds'] + 2100)
    deadline=time.monotonic()+wait_seconds
    try:
        atomic_json(args.output/'status.json',{'phase':'waiting_for_training_exit'})
        while not (args.run/'exit.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('No source exit receipt within bounded wait')
            time.sleep(15)
        assert json.loads((args.run/'exit.json').read_text())['exit_code']==0
        assert json.loads((args.run/'completed.json').read_text())['status']=='completed'
        config=json.loads((args.run/'config.json').read_text())
        commands=[('onnx',False,[sys.executable,'-u','-B','tools/export_b0_onnx.py','--run',str(args.run),'--output',str(args.output/'onnx')]),
                  ('diagnostics',False,[sys.executable,'-u','-B','tools/audit_b0_errors.py','--run',str(args.run),'--output',str(args.output/'error_diagnostics.json')]),
                  ('comparison',True,[sys.executable,'-u','-B','tools/compare_night_models.py','--run',str(args.run),'--baseline',str(args.baseline),'--output',str(args.output/'comparison')])]
        for name,gpu,command in commands:
            if gpu:
                active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
                if config['gpu_uuid'] in active:raise RuntimeError('GPU0 occupied after training; comparison deferred, do not stop other work')
            atomic_json(args.output/'status.json',{'phase':name})
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=config['gpu_uuid'] if gpu else '')
            atomic_json(args.output/(name+'_command.json'),{'command':command,'cwd':str(code),'gpu':gpu})
            began=time.monotonic()
            with (args.output/(name+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            atomic_json(args.output/(name+'_exit.json'),{'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-began})
            if result.returncode:raise RuntimeError(name+' failed; see log')
        subprocess.run([sys.executable,'-B','tools/build_training_review.py','--run',str(args.run)],cwd=code,check=True)
        index=json.loads((args.run/'checkpoint_index.json').read_text())
        for entry in index['checkpoints']:assert sha(args.run/entry['path'])==entry['sha256']
        verification={'status':'passed','checkpoints_verified':len(index['checkpoints']),
                      'data_lock_sha256':validate_data_lock(config['data_root'],code),
                      'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip()}
        atomic_json(args.output/'verification.json',verification)
        atomic_json(args.output/'status.json',{'phase':'completed'})
        keep={'checkpoints/best.pt','checkpoints/last.pt',index['best']['path'],index['last']['path']}
        files=[]
        for path in sorted(args.run.rglob('*')):
            if not path.is_file():continue
            relative=str(path.relative_to(args.run))
            if relative.startswith('checkpoints/') and relative not in keep:continue
            if relative.startswith('evaluation/') and path.name!='metrics.json':continue
            files.append(path)
        files+=sorted(p for p in args.output.rglob('*') if p.is_file())
        host_receipt=args.run.parent/(args.run.name+'-launch_host.json')
        if host_receipt.exists():files.append(host_receipt)
        # This process logs outside output, so every archived file is now stable.
        manifest=args.run.parent/(args.run.name+'-delivery-manifest.json')
        atomic_json(manifest,{str(p.relative_to(args.run.parent)):sha(p) for p in files})
        files.append(manifest)
        archive=args.run.parent/(args.run.name+'-delivery.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for path in files:tar.add(path,arcname=str(path.relative_to(args.run.parent)))
        atomic_json(args.run.parent/(args.run.name+'-delivery.json'),{'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)})
        print('Night run delivery archive ready',str(archive),flush=True)
    except Exception:
        atomic_json(args.output/'status.json',{'phase':'failed','error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
