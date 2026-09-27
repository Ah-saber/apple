"""Conventional affine8 input-activation pressure; not a prediction of SDK mixed precision."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');r=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
l=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);results={};manifest=json.loads((r/'calibration_inputs'/f'{a.scene}_manifest.json').read_text())
with torch.inference_mode():
 for case in a.cases:
  m=load_deployment(a.scene,case,r);ranges={};hooks=[]
  def hook(name):
   def capture(module,args):
    x=args[0].float();lo=float(x.min());hi=float(x.max());prior=ranges.get(name,{'min':0.,'max':0.,'input_shape':list(x.shape),'module':type(module).__name__});prior['min']=min(prior['min'],lo);prior['max']=max(prior['max'],hi);ranges[name]=prior
   return capture
  for name,layer in m.named_modules():
   if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):hooks.append(layer.register_forward_pre_hook(hook(name)))
  sources=[]
  for item in manifest['inputs']:
   file=r/'calibration_inputs'/item['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'];v=np.load(file);x=torch.from_numpy(v['nine_raw']).cuda();ctx=torch.from_numpy(v['reference_thumb']).cuda();x,ctx=prepare_inputs(m,x,ctx);y=m(x,ctx);assert torch.isfinite(y).all();sources.append({'file':item['file'],'sha256':item['sha256'],'sample_id':item['sample_id']})
  for h in hooks:h.remove()
  results[case]={'ranges':ranges,'sources':sources,'GT_used':False,'calibration_split':'training records; test labels unused'};del m,x,ctx,y;torch.cuda.empty_cache();print('ACTIVATION_CALIBRATION',a.scene,case,len(ranges),flush=True)
(r/f'{a.scene}_activation_calibration_ranges.json').write_text(json.dumps({'scene':a.scene,'results':results,'SDK_mixed_precision_unknown':True,'ranges_from_four_training_input_vectors':True},indent=2))
