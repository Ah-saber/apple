"""Isolated weight8 stress controls; do not reproduce SDK activation quantization."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from continuation_candidates import load_continuation as load_structural
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927'
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
def quant(w,mode):
 v=w.float();scale=(v.abs().amax(tuple(range(1,v.ndim)),keepdim=True) if mode=='per_output' else v.abs().max()).clamp_min(1e-9)/127
 return (v/scale).round().clamp(-127,127)*scale
rows=[]
with torch.inference_mode():
 for scene,frame in [('ordinary',20),('special',60)]:
  v=np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda()
  for case,scope in [('baseline','output'),('phase3_preserve_nearest','output'),('phase3_quantsearch_nearest','output'),('baseline','reference'),('reference_relu_output_stable','reference'),('space_pack_reference_relu_output_stable_phase3_preserve_nearest','reference_and_output')]:
   floating=load_structural(scene,case,out)(x,c).float()
   for mode in ['per_output','per_tensor']:
    model=load_structural(scene,case,out)
    if 'reference' in scope:
     for layer in model.core.model.global_reference.modules():
      if isinstance(layer,torch.nn.Conv2d):layer.weight.copy_(quant(layer.weight,mode))
    if 'output' in scope:
     layer=model.output.conv if hasattr(model,'output') and model.output is not None else model.core.model.upsample[0];layer.weight.copy_(quant(layer.weight,mode))
    y=model(x,c).float();delta=(y-floating).abs();pooled=torch.nn.functional.avg_pool2d(y-floating,3,3).abs();rows.append({'scene':scene,'case':case,'scope':scope,'assumed_weight8_rule':mode,'full_mean_gray':float(delta.mean()),'full_max_gray':float(delta.max()),'native_mean_gray':float(pooled.mean()),'native_max_gray':float(pooled.max()),'SDK_verified':False,'activation16_simulated':False})
    print('QUANT',rows[-1],flush=True)
(out/'weight8_searched_stress.json').write_text(json.dumps({'assumptions':'symmetric signed127; weights only; two assumed rules; no SDK equivalence','checks':rows},indent=2))
