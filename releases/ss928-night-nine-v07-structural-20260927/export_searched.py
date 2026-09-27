"""Incremental exports of calibrated nine-phase output, plus native-path rewrites."""
import argparse,fcntl,gc,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();base=Path(__file__).parent
sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages']
import onnx
from structural_candidates import load_structural
from materialize_aliases import materialize
from rewrite_hotspots import rewrite_statistics
from rewrite_reference_relu import rewrite as rewrite_ref
from reduce_output_phases import reduce_phases
root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-V07-STRUCTURAL-20260927';folder=run/'source_onnx'
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cases=['phase3_quantsearch_nearest','phase3_quantsearch_fixed','space_pack_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_fixed'];rows=[]
for case in cases:
 m=load_structural(a.scene,case,run)
 for small in (True,False):
  h,w=(64,96) if small else (1024,1280);suffix='_small' if small else '';path=folder/f'{a.scene}_{case}{suffix}.onnx'
  if path.exists():raise FileExistsError(path)
  x=torch.full((1,9,h,w),.3,device='cuda');c=torch.full((1,1,64,64),.3,device='cuda')
  torch.onnx.export(m,(x,c),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
  g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path)
  rows.append({'file':path.name,'case':case,'small':small,'SDK_verified':False});print('EXPORT',path.name,flush=True)
 del m;gc.collect();torch.cuda.empty_cache()
for small in (True,False):
 suffix='_small' if small else '';old=onnx.load(folder/f'{a.scene}_baseline{suffix}.onnx');projection=onnx.load(folder/f'{a.scene}_phase3_quantsearch_nearest{suffix}.onnx')
 for case in cases:
  g=old
  if case.startswith('space_pack'):g=rewrite_statistics(g,'space_pack')
  if 'reference_relu' in case:g,_=rewrite_ref(g,old,onnx.load(folder/f'{a.scene}_reference_relu_output_stable{suffix}.onnx'))
  g=reduce_phases(g,3,'preserve_fixed' if case.endswith('fixed') else 'preserve_nearest',projection,old)
  path=folder/f'{a.scene}_{case}_rewritten{suffix}.onnx'
  if path.exists():raise FileExistsError(path)
  onnx.save(g,path);rows.append({'file':path.name,'case':case,'small':small,'bounded_rewrite':True,'SDK_verified':False});print('REWRITE',path.name,flush=True)
(run/f'{a.scene}_searched_export_manifest.json').write_text(json.dumps({'models':rows,'NPU_measured':False},indent=2))
