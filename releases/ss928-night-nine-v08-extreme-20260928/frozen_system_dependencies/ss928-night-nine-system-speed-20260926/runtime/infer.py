import argparse,fcntl,json
from pathlib import Path
import numpy as np
import torch
from load_system_release import load_release,prepare_inputs
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--case',default='primary');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--base-dir',type=Path);p.add_argument('--execution',choices=['source','compiled'],default='source');p.add_argument('--gpu-lock',type=Path);a=p.parse_args()
if a.gpu_lock:lease=a.gpu_lock.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(torch.device(a.device)).total_memory)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,info=load_release(a.scene,a.case,a.device,base_dir=a.base_dir)
with np.load(a.input,allow_pickle=False) as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw']));c=torch.from_numpy(np.ascontiguousarray(v['reference_thumb']))
if not torch.isfinite(x).all() or not torch.isfinite(c).all():raise ValueError('Nonfinite input')
x,c=prepare_inputs(x,c,info,a.device)
if a.execution=='compiled':model=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
with torch.inference_mode():result=model(x,c).cpu().numpy()
if list(result.shape)!=info['output_shape']:raise ValueError('Incorrect output shape')
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('xb') as f:np.savez_compressed(f,display_gray=result)
print(json.dumps({'IO':info,'execution':a.execution,'output':str(a.output),'NPU_verified':False},indent=2))
