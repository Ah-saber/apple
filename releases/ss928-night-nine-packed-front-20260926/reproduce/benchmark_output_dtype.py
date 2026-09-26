"""Paired output-precision timing for chosen complete models; same FP32 input."""
import fcntl,json,sys,gc
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from load_packed_model import load_packed_model
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
report={'input_dtype':'float32','output_shape':[1,1,3072,3840],'warmup':30,'rounds':3,'samples_per_round':200,'CUDA_graphs':False,'TF32':False,'NPU_verified':False,'results':{}}
for scene,frame in [('ordinary',20),('special',60)]:
 run=root/'runs/SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else out;step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();c=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={}
 for dtype in ['float32','float16']:
  m=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp,output=dtype);torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
  with torch.inference_mode():
   source=m(x,c);value=exe(x,c);d=(source.float()-value.float()).abs();num={'compiled_vs_source_max_gray':float(d.max()),'compiled_vs_source_mean_gray':float(d.mean())}
   if dtype=='float32':expected=source.cpu().numpy()
   else:
    actual=source.cpu().numpy();assert np.array_equal(actual,expected.astype(np.float16));num['rounding_max_gray']=float(np.abs(actual.astype(np.float32)-expected).max());num['rounded_uint8_changed_pixels']=int((np.rint(actual.astype(np.float32))!=np.rint(expected)).sum());num['total_pixels']=int(actual.size)
   del source,value,d
   for _ in range(30):exe(x,c)
   torch.cuda.synchronize();rounds=[]
   for _ in range(3):
    times=[]
    for _ in range(200):
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();exe(x,c);e.record();e.synchronize();times.append(s.elapsed_time(e))
    rounds.append(times)
  flat=sum(rounds,[]);r={'mean_ms':float(np.mean(flat)),'p95_ms':float(np.percentile(flat,95)),'samples_ms':rounds,'numerical':num};report['results'][scene][dtype]=r;print(scene,dtype,{k:v for k,v in r.items() if k!='samples_ms'},flush=True);del exe,m;gc.collect();torch.cuda.empty_cache()
 (out/'output_dtype_gpu_timing.json').write_text(json.dumps(report,indent=2));del x,c
