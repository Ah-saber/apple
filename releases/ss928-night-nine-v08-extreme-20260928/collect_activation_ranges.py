"""Actual full-vector activation ranges; outside all timing measurements."""
import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);frame=20 if a.scene=='ordinary' else 60;v=np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{a.scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();ctx=torch.from_numpy(v['reference_thumb']).cuda();results={}
with torch.inference_mode():
 for case in a.cases:
  model=load_deployment(a.scene,case,run);records=[];hooks=[]
  def hook(name):
   def capture(module,args,value):
    y=value.float();lo=y.amin((0,2,3));hi=y.amax((0,2,3));records.append({'module':name,'shape':list(value.shape),'dtype':str(value.dtype),'finite':bool(torch.isfinite(value).all()),'per_channel_min':lo.cpu().tolist(),'per_channel_max':hi.cpu().tolist(),'max_abs':float(y.abs().max())})
   return capture
  for name,module in model.named_modules():
   if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):hooks.append(module.register_forward_hook(hook(name)))
  xx,cc=prepare_inputs(model,x,ctx);output=model(xx,cc)
  for h in hooks:h.remove()
  assert torch.isfinite(output).all();results[case]={'declared_input_shape':list(xx.shape),'input_dtype':str(xx.dtype),'full_output_shape':list(output.shape),'output_dtype':str(output.dtype),'activations':records,'timing_diagnostic_only':True};del model,output;torch.cuda.empty_cache()
(run/f'{a.scene}_activation_ranges.json').write_text(json.dumps({'scene':a.scene,'frame':frame,'GT_used':False,'activation_quantization_simulated':False,'results':results},indent=2))
