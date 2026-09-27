"""Full source graphs, deployment rewrites, and small independent controls."""
import argparse,fcntl,json,sys,gc
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
base=Path(__file__).parent;sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages']
import onnx
from structural_candidates import load_structural
from materialize_aliases import materialize
from rewrite_hotspots import rewrite_statistics,rewrite_output
from reduce_output_phases import reduce_phases
from rewrite_reference_relu import rewrite as rewrite_reference
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-STRUCTURAL-20260927';(out/'source_onnx').mkdir(exist_ok=True)
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
cases=['baseline','space_pack','temporal_space_pack','axis_horizontal','axis_vertical','phase3_nearest','phase3_preserve_nearest','phase3_preserve_fixed','phase3_bilinear','phase3_fixed','phase2_nearest','phase1_nearest','reference_relu','reference_relu_output_stable','phase3_k1_nearest','phase3_k1_ridge_nearest','phase3_preserve_nearest_fp32','space_pack_phase3_nearest','space_pack_phase3_preserve_nearest','space_pack_reference_relu_output_stable','space_pack_reference_relu_output_stable_phase3_preserve_nearest'];manifest=[]
for case in cases:
 model=load_structural(a.scene,case,out)
 for small in [True,False]:
  h,w=(64,96) if small else (1024,1280);x=torch.full((1,9,h,w),.3,device='cuda');ctx=torch.full((1,1,64,64),.3,device='cuda');path=out/'source_onnx'/f'{a.scene}_{case}{"_small" if small else ""}.onnx'
  if path.exists():raise FileExistsError(path)
  torch.onnx.export(model,(x,ctx),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path)
  manifest.append({'file':str(path),'case':case,'small':small,'input_shape':[1,9,h,w],'output_shape':[1,1,h*3,w*3],'SDK_verified':False});print('EXPORTED',path.name,flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
for small in [True,False]:
 suffix='_small' if small else '';old=onnx.load(out/'source_onnx'/f'{a.scene}_baseline{suffix}.onnx')
 for case in cases[1:]:
  if case in ['space_pack','temporal_space_pack']:g=rewrite_statistics(old,case)
  elif case.startswith('axis_'):g=rewrite_output(old,case)
  elif case.startswith('phase'):
   factor=int(case[5]);mode=case.split('_',1)[1]
   if 'k1' in case or 'fp32' in case:
    continue
   g=reduce_phases(old,factor,mode)
  elif 'space_pack' in case:
   g=rewrite_statistics(old,'space_pack')
   if 'reference_relu' in case:g,_=rewrite_reference(g,old,onnx.load(out/'source_onnx'/f'{a.scene}_reference_relu_output_stable{suffix}.onnx'))
   if 'phase3_' in case:g=reduce_phases(g,3,'preserve_nearest' if 'preserve_nearest' in case else ('bilinear' if case.endswith('bilinear') else 'nearest'))
  else:g,_=rewrite_reference(old,old,onnx.load(out/'source_onnx'/f'{a.scene}_{case}{suffix}.onnx'))
  path=out/'source_onnx'/f'{a.scene}_{case}_rewritten{suffix}.onnx';onnx.checker.check_model(g);onnx.save(g,path);manifest.append({'file':str(path),'case':case,'small':small,'bounded_rewrite':True,'SDK_verified':False});print('REWRITTEN',path.name,flush=True)
(out/f'{a.scene}_export_manifest.json').write_text(json.dumps({'models':manifest,'NPU_measured':False},indent=2))
