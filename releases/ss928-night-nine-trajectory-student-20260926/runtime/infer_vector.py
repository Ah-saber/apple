"""Infer already-preprocessed nine-frame vectors and save effective uint8 output."""
import argparse
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from load_candidate import load_candidate
parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True,type=Path)
parser.add_argument('--baseline',required=True,type=Path)
parser.add_argument('--input',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--execution',choices=('source','compiled'),default='compiled')
parser.add_argument('--device',default='cuda')
parser.add_argument('--input-precision',choices=('float32','float16'),default='float32')
parser.add_argument('--gpu-lock',type=Path)
parser.add_argument('--memory-limit-gib',type=float,default=2.5)
args=parser.parse_args()
lease=None
if args.gpu_lock:
    lease=args.gpu_lock.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
if args.device.startswith('cuda'):
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*1024**3/torch.cuda.get_device_properties(torch.device(args.device)).total_memory)
with np.load(args.input,allow_pickle=False) as vectors:
    stack=np.ascontiguousarray(vectors['nine_raw'],dtype=np.float16 if args.input_precision=='float16' else np.float32)
    context=np.ascontiguousarray(vectors['reference_thumb'],dtype=np.float32)
if stack.shape!=(1,9,1024,1280) or context.shape!=(1,1,64,64):raise ValueError('Invalid full-frame input shapes')
if not np.isfinite(stack).all() or not np.isfinite(context).all():raise ValueError('Non-finite input')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
model=load_candidate(args.model,args.baseline,args.device,args.execution,remove_zero_init=True)
x=torch.from_numpy(stack).to(args.device);thumb=torch.from_numpy(context).to(args.device)
with torch.inference_mode():
    result=model(x,thumb).cpu().numpy()
if result.shape!=(1,1,3072,3840) or result.dtype!=np.uint8:raise ValueError('Unexpected model output')
with args.output.open('xb') as destination:destination.write(result.tobytes())
print(json.dumps({'output':str(args.output),'shape':list(result.shape),'dtype':str(result.dtype),'bytes':int(result.nbytes)},indent=2))
