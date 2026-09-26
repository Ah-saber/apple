"""Independent complete nine-frame model execution from a release directory."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));p.add_argument('--variant',default='selected',choices=('selected','joint8','joint12','s8'));p.add_argument('--dtype',default='float32',choices=('float32','float16'));p.add_argument('--execution',default='source',choices=('source','compiled'));p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock-file',type=Path);p.add_argument('--gpu-memory-gib',type=float,default=2.5);a=p.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'runtime'))
from load_packed_model import load_packed_model
from load_student_model import load_student_model
from output_variant import HalfShuffleStudent
if a.lock_file:
 import fcntl
 lease=a.lock_file.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(a.gpu_memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
variant=('joint8' if a.scene=='ordinary' else 'joint12') if a.variant=='selected' else a.variant
cp=root/'models'/f'{a.scene}_factor24.pt';bp=root/'models'/f'{a.scene}_fused.pt'
if variant=='s8':
 s=load_student_model(root/'models'/f'{a.scene}_student_compact8.pt',cp,bp);m=HalfShuffleStudent(s.core,s.student,output=a.dtype)
else:m=load_packed_model(root/'models'/f'{a.scene}_packed_{variant}.pt',cp,bp,output=a.dtype)
if a.execution=='compiled':
 torch.compiler.reset();m=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
with np.load(a.input,allow_pickle=False) as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();c=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
assert tuple(x.shape)==(1,9,1024,1280) and tuple(c.shape)==(1,1,64,64)
with torch.inference_mode():y=m(x,c).cpu().numpy()
assert y.shape==(1,1,3072,3840) and np.isfinite(y).all()
a.output.parent.mkdir(parents=True,exist_ok=True);y.astype('<f2' if a.dtype=='float16' else '<f4').tofile(a.output)
r={'scene':a.scene,'variant':variant,'execution':a.execution,'shape':list(y.shape),'dtype':str(y.dtype),'byte_count':y.nbytes,'output':str(a.output)};a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
