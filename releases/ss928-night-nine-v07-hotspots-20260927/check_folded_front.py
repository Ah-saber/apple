import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from operator_candidates import CandidateSystem,FoldedStatisticsFront
sys.path.insert(0,'/tmp/ss928_release_verification_20260926/ss928-night-nine-system-speed-20260926/runtime')
from load_system_release import load_release
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
lock=Path('/data/zhangbenzhuang/huawei_sr/runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base,_=load_release(a.scene,'front_f32');frame=20 if a.scene=='ordinary' else 60
v=np.load(f'/data/zhangbenzhuang/huawei_sr/runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS/{a.scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda();report=[]
with torch.inference_mode():
 expected=base(x,c)
 for precision in ('float16','float32'):
  model=CandidateSystem(base);model.front=FoldedStatisticsFront(base.front,precision);out=model(x,c);d=(out-expected).abs();torch.compiler.reset();torch._inductor.config.triton.cudagraphs=False;compiled=torch.compile(model,fullgraph=True)
  for _ in range(30):compiled(x,c)
  times=[]
  for _ in range(3):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(200):compiled(x,c)
   end.record();end.synchronize();times.append(start.elapsed_time(end)/200)
  item={'precision':precision,'mean_gray':float(d.mean()),'max_gray':float(d.max()),'rounded_byte_max_error':float((out.round()-expected.round()).abs().max()),'compiled_mean_ms':float(np.mean(times)),'three_batch_means_ms':times,'NPU_measured':False};report.append(item);print(a.scene,item,flush=True)
Path(f'/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-FOLLOWUP-20260927/{a.scene}_folded_front.json').write_text(json.dumps(report,indent=2))
