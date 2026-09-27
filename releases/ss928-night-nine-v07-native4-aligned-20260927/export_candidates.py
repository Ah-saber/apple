"""Export full-source continuation candidates and bounded small controls."""
import argparse,fcntl,json,sys,gc
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tag',default='export_manifest');p.add_argument('--cases',nargs='+',required=True);a=p.parse_args()
base=Path(__file__).parent;sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnx
from continuation_candidates import load_continuation,prepare_inputs
from materialize_aliases import materialize
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927';(out/'source_onnx').mkdir(exist_ok=True);lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=[]
for case in a.cases:
 m=load_continuation(a.scene,case,out)
 for small in (True,False):
  h,w=(64,96) if small else (1024,1280);path=out/'source_onnx'/f'{a.scene}_{case}{"_small" if small else ""}.onnx'
  if path.exists():raise FileExistsError(path)
  x=torch.full((1,9,h,w),.3,device='cuda');c=torch.full((1,1,64,64),.3,device='cuda')
  x,c=prepare_inputs(m,x,c)
  torch.onnx.export(m,(x,c),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path);rows.append({'file':path.name,'case':case,'small':small,'SDK_verified':False});print('EXPORTED',path.name,flush=True)
 del m;gc.collect();torch.cuda.empty_cache()
(out/f'{a.scene}_{a.tag}.json').write_text(json.dumps({'models':rows,'NPU_verified':False},indent=2))
