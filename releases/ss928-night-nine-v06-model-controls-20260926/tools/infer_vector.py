"""Reproduce source candidates using the preserved v0.6 release weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--variant',required=True,choices=['base','two_three','three_two','fixed_pack','fixed_two_three','dither_one','dither_two','dither_fixed']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-release',type=Path);p.add_argument('--lock-file',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=a.base_release or root.parent/'ss928-night-nine-packed-front-20260926';sys.path[:0]=[str(root/'runtime'),str(base/'runtime')]
from load_packed_model import load_packed_model
from model_variants import CompleteVariant,MeanRawDitherDisplay
if a.lock_file:
 import fcntl
 lease=a.lock_file.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
width='joint8' if a.scene=='ordinary' else 'joint12';models=base/'models';m=load_packed_model(models/f'{a.scene}_packed_{width}.pt',models/f'{a.scene}_factor24.pt',models/f'{a.scene}_fused.pt')
if a.variant.startswith('dither'):
 if a.variant=='dither_fixed':m=CompleteVariant(m,fixed_pack=True)
 m=MeanRawDitherDisplay(m,two_stage=a.variant=='dither_two')
elif a.variant!='base':
 order=(3,2) if a.variant=='three_two' else ((2,3) if a.variant in ['two_three','fixed_two_three'] else (6,1));m=CompleteVariant(m,first=order[0],second=order[1],fixed_pack=a.variant.startswith('fixed'))
with np.load(a.input,allow_pickle=False) as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
assert tuple(x.shape)==(1,9,1024,1280) and tuple(ctx.shape)==(1,1,64,64)
with torch.inference_mode():y=m(x,ctx).cpu().numpy()
assert y.shape==(1,1,3072,3840) and y.dtype==np.float32 and np.isfinite(y).all()
a.output.parent.mkdir(parents=True,exist_ok=True);y.astype('<f4').tofile(a.output);print(json.dumps({'scene':a.scene,'variant':a.variant,'execution':'source','output':str(a.output),'shape':list(y.shape),'dtype':str(y.dtype),'bytes':y.nbytes},indent=2))
