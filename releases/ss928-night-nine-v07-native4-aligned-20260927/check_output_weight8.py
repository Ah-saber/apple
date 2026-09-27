"""Output-only symmetric weight8 stress, not the SDK quantization pipeline."""
import argparse,copy,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-V07-NATIVE4-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from continuation_candidates import load_continuation
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
f=20 if a.scene=='ordinary' else 60;v=np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{a.scene}_frame_{f}.npz');x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda();rows=[]
def quant(w,axis):
 z=w.float();dims=tuple(i for i in range(z.ndim) if i!=axis) if axis is not None else tuple(range(z.ndim));scale=z.abs().amax(dims,keepdim=True).clamp_min(1e-12)/127
 return (z/scale).round().clamp(-127,127).mul(scale).to(w.dtype)
with torch.inference_mode():
 for case in ('previous_combo','combo_aligned16_nearest','combo_aligned16_fixed','combo_tail5_exact_edges','native4_linear_trained','native4_linear_dither_trained','native4_linear_dither_float_trained','native4_relu8_ref_dither_trained'):
  m=load_continuation(a.scene,case,run);target=m(x,c).float();output=m.output
  for mode in ('per_tensor','per_output'):
   q=copy.deepcopy(m);changes=[]
   for name,module in q.output.named_modules():
    if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
     axis=1 if isinstance(module,torch.nn.ConvTranspose2d) else 0
     module.weight.copy_(quant(module.weight,axis if mode=='per_output' else None));changes.append(name+'.weight')
   for name,module in q.output.named_modules():
    for key in ('kernel','weight'):
     z=module._buffers.get(key)
     if z is not None and z.ndim==4:
      z.copy_(quant(z,1 if mode=='per_output' else None));changes.append(name+'.'+key)
   value=q(x,c).float();d=(value-target).abs();rows.append({'case':case,'mode':mode,'output_only_weight_buffers':changes,'mean_gray':float(d.mean()),'max_gray':float(d.max()),'rounded_native_mean_delta_gray':float((torch.nn.functional.avg_pool2d(value.clamp(0,255).round(),3,3)-torch.nn.functional.avg_pool2d(target.clamp(0,255).round(),3,3)).abs().mean())});print('WEIGHT8',a.scene,rows[-1],flush=True);del q
  del m,target
(run/f'{a.scene}_output_weight8.json').write_text(json.dumps({'scene':a.scene,'activation_quantization':False,'bias_quantization':False,'SDK_verified':False,'controls':rows},indent=2))
