"""Independent source inference of the complete learned-trajectory model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));p.add_argument('--variant',default='compact8',choices=('compact8','wide12'));p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execution',choices=('source','compiled'),default='source');p.add_argument('--lock-file',type=Path);p.add_argument('--gpu-memory-gib',type=float,default=2.5);a=p.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'runtime'))
from load_student_model import load_student_model
if a.lock_file:
 import fcntl
 lease=a.lock_file.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.cuda.set_per_process_memory_fraction(a.gpu_memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=load_student_model(root/'models'/f'{a.scene}_student_{a.variant}.pt',root/'models'/f'{a.scene}_factor24.pt',root/'models'/f'{a.scene}_fused.pt',execution=a.execution)
with np.load(a.input,allow_pickle=False) as v:
 x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();c=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
assert tuple(x.shape)==(1,9,1024,1280) and tuple(c.shape)==(1,1,64,64)
with torch.inference_mode():y=m(x,c).float().cpu().numpy()
assert y.shape==(1,1,3072,3840) and np.isfinite(y).all()
a.output.parent.mkdir(parents=True,exist_ok=True);y.astype('<f4').tofile(a.output)
record={'scene':a.scene,'variant':a.variant,'execution':a.execution,'shape':list(y.shape),'dtype':'float32','byte_count':y.nbytes,'output':str(a.output)}
a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
