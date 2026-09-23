"""Shared5090 full-frame FP32 inference timings, identical complete neural interfaces."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
C=Path(__file__).resolve().parents[1];sys.path.insert(0,str(C/'src'))
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
from ir_sr.training import dataset_for_config,atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
R=Path('/data/zhangbenzhuang/huawei_sr/runs');lease=(R/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
entries={v['key']:v for v in json.loads(a.models.read_text())['models']}
runs={k:Path(v['checkpoint']) for k,v in entries.items()}
before=subprocess.check_output(['nvidia-smi'],text=True);rows=[];module_rows=[];result={}
def summary(v):return dict(n=len(v),mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(np.max(v)))
for key in list(runs):runs[key+'_fastpool']=runs[key]
for tag,run in runs.items():
 s=torch.load(run,map_location='cpu',weights_only=False);c=s['config'];model=FullFrameStudent(inference_model(c,s['model']),hierarchical_pooling=tag.endswith('fastpool')).cuda().eval();del s
 ds=dataset_for_config(c,'val');x=ds.full_raw(0)[None].cuda()
 with torch.inference_mode():
  for _ in range(50):y=model(x)
  torch.cuda.synchronize()
  times=[]
  for repeat in range(3):
   for index in range(200):
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record();y=model(x);end.record();end.synchronize();elapsed=begin.elapsed_time(end)
    rows.append(dict(model=tag,repeat=repeat,index=index,milliseconds=elapsed));times.append(elapsed)
  # Profile leaf stages separately; hooks affect timing, so keep apart from normal benchmark.
  samples={};handles=[];events={}
  modules={'thumbnail_pool':model.thumbnail_pool,'reference_pool':model.reference_pool,'reference_resize':model.reference_resize,'input_rearrange':model.model.down,'head_conv':model.model.head[0],'tail_norm':model.model.tail[0],'tail_conv':model.model.tail[1],'output_conv':model.model.upsample[0],'reference_project':model.model.global_reference.project}
  for i,block in enumerate(model.model.body):
   modules.update({f'body{i}_norm':block.norm,f'body{i}_conv':block.conv1,f'body{i}_activation':block.act})
  for i,module in enumerate(model.model.upsample[1:]):modules[f'shuffle_stage{i}']=module
  for i,module in enumerate(model.model.global_reference.encoder):modules[f'reference_encoder{i}']=module
  for name,module in modules.items():
   def pre(m,args,n=name):
    e=torch.cuda.Event(enable_timing=True);e.record();events[n]=[e,None]
   def post(m,args,out,n=name):
    e=torch.cuda.Event(enable_timing=True);e.record();events[n][1]=e
   handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
  for i in range(20):
   y=model(x);torch.cuda.synchronize()
   for n,(begin,end) in events.items():
    elapsed=begin.elapsed_time(end);samples.setdefault(n,[]).append(elapsed);module_rows.append(dict(model=tag,index=i,module=n,milliseconds=elapsed))
  for handle in handles:handle.remove()
 result[tag]=dict(total=summary(times),modules={n:summary(v) for n,v in samples.items()},checkpoint_sha256=sha(run))
 del model,x,y,ds;torch.cuda.empty_cache();print(tag,json.dumps(result[tag]['total']),flush=True)
for name,values in [('per_frame.csv',rows),('module_times.csv',module_rows)]:
 with (a.output/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
atomic_json(a.output/'report.json',dict(status='complete',results=result,gpu_before=before,gpu_after=subprocess.check_output(['nvidia-smi'],text=True),protocol='Shared5090, FP32 fused whole-frame graph, TF32 off, batch1, RAW1024x1280->3072x3840; thumbnail/reference included; excludes offline calibration, H2D/D2H, uint8 and file IO.50warmup+3x200 CUDA events synchronized.20separate module samples. Feature additions included in total; disjoint module intervals omit small host scheduling and additions. Concurrent load makes these diagnostic, not exclusive hardware acceptance. SS928 UNTESTED.',torch=torch.__version__))
