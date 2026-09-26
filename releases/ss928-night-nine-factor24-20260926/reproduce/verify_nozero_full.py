"""Compare the original trained nine-frame model and fastest candidate against identical GT."""
import argparse
import fcntl
import json
import math
import sys
import hashlib
import subprocess
import types
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser();p.add_argument('--scene',choices=('night_ordinary','night_special'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=24);a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');special=a.scene=='night_special'
code=root/'code/worktrees'/('ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925')
ck=root/'runs'/('SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1/checkpoints/step_000002000.pt' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1/checkpoints/step_000002000.pt')
sys.path.insert(0,str(code/'src'))
from ir_sr.model import inference_model,to_deploy
from ir_sr.training import dataset_for_config,sha
from ir_sr.metrics import image_metrics
from trajectory_half_trimmed import trajectory_features_half_trimmed
from static_nine_uint8_half import StaticNineByteHalf
from collapse_output_shuffles import collapse_output_shuffles
from fused_nine import FusedNine
from trajectory_batched_stats import trajectory_features_batched_stats
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
s=torch.load(ck,map_location='cpu',weights_only=False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'));sys.path.insert(0,'/tmp')
from runtime_nine import load_model
from refine_variants import FactorizedFirst
from load_candidate import load_candidate

models=[]
for nozero in (False,True):
 wrapper=load_candidate(root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/('special_factor24.pt' if special else 'ordinary_factor24.pt'),root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/('special_fused.pt' if special else 'ordinary_fused.pt'),remove_zero_init=nozero)
 models.append(torch.compile(wrapper,fullgraph=True,options={'triton.cudagraphs':False}))
ds=dataset_for_config(s['config'],'test');rec=next(r for r in ds.records if r['scene_id']==a.scene)
count=120 if special else 60;diffs=[]
with torch.inference_mode():
 for frame in range(count):
  row=dict(rec,frame_id=frame);stack=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda().half()
  context,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));context=context[None].cuda()
  before,after=models[0](stack,context),models[1](stack,context)
  difference=(before.to(torch.int16)-after.to(torch.int16)).abs()
  item={'frame':frame,'max_gray':int(difference.max()),'changed_pixels':int(torch.count_nonzero(difference))}
  diffs.append(item);assert item['changed_pixels']==0,item
  if frame%10==0:print(a.scene,frame,'exact',flush=True)
a.output.write_text(json.dumps({'scene':a.scene,'frames':count,'full_output_shape':[1,1,3072,3840],'bit_identical':True,'differences':diffs},indent=2))
