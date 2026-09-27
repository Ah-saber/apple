import argparse,fcntl,gc,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from load_system_release import load_release,prepare_inputs
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--vector',type=Path,required=True);p.add_argument('--cases',nargs='+',default=['base','front_f32','primary']);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-lock',type=Path);a=p.parse_args()
if a.gpu_lock:lease=a.gpu_lock.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
with np.load(a.vector,allow_pickle=False) as v:stack=torch.from_numpy(np.ascontiguousarray(v['nine_raw']));context=torch.from_numpy(np.ascontiguousarray(v['reference_thumb']))
report={'scene':a.scene,'GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'warmup':30,'rounds':3,'samples_per_round':200,'TF32':False,'CUDA_graphs':False,'timed_input_preparation':False,'NPU_verified':False,'results':{}}
for case in a.cases:
 model,info=load_release(a.scene,case);x,c=prepare_inputs(stack,context,info,'cuda');torch.compiler.reset();compiled=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
 with torch.inference_mode():
  source=model(x,c).float();actual=compiled(x,c).float();delta=(source-actual).abs();numerical={'mean_gray':float(delta.mean()),'max_gray':float(delta.max())};del source,actual,delta
  for _ in range(30):compiled(x,c)
  torch.cuda.synchronize();rounds=[]
  for _ in range(3):
   samples=[]
   for _ in range(200):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();compiled(x,c);end.record();end.synchronize();samples.append(start.elapsed_time(end))
   rounds.append(samples)
 flat=sum(rounds,[]);report['results'][case]={'IO':info,'mean_ms':float(np.mean(flat)),'p95_ms':float(np.percentile(flat,95)),'samples_ms':rounds,'compiled_vs_source':numerical};print(case,report['results'][case]['mean_ms'],flush=True);del model,compiled,x,c;gc.collect();torch.cuda.empty_cache()
a.output.write_text(json.dumps(report,indent=2))
