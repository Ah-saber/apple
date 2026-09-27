"""Run complete packaged source models using their declared interface."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--release',default=str(Path(__file__).parent));p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--case',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();r=Path(a.release);sys.path.insert(0,str(r/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
out=Path(a.output)
if out.exists():raise FileExistsError(out)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;v=np.load(a.input);model=load_deployment(a.scene,a.case,r,device=a.device);x=torch.from_numpy(v['nine_raw']).to(a.device);c=torch.from_numpy(v['reference_thumb']).to(a.device);x,c=prepare_inputs(model,x,c)
with torch.inference_mode():y=model(x,c).cpu().numpy()
assert np.isfinite(y).all();np.savez_compressed(out,output=y);print(out,y.shape,y.dtype)
