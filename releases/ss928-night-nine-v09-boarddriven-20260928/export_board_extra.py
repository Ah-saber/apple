"""Exact full-geometry export and real-input source proof for board layout controls."""
import fcntl,hashlib,json,sys,gc
from pathlib import Path
import numpy as np
import torch
RUN=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V09-20260928');sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parent/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnx
from board_candidates import load_board,prepare_inputs
from materialize_aliases import materialize
lease=(RUN.parent/'TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cases=['v08_c11','rows16static_refafter','rows32_refafter','rows16static_meanfirst','raw6_rows16static','raw6_rows32'];(RUN/'source_onnx').mkdir(exist_ok=True);(RUN/'reference_outputs').mkdir(exist_ok=True);rows=[]
for scene,frame in [('ordinary',20),('special',60)]:
 v=np.load(RUN/'v08/test_vectors'/f'{scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda();baseline=None
 for case in cases:
  m=load_board(scene,case,RUN);xx,cc=prepare_inputs(m,x,c)
  with torch.inference_mode():y=m(xx,cc).cpu().numpy()
  assert tuple(y.shape)==(1,1,3072,3840) and np.isfinite(y).all()
  if case=='v08_c11':baseline=y
  d=np.abs(y.astype(np.float32)-baseline.astype(np.float32))
  if case!='v08_c11':np.savez_compressed(RUN/'reference_outputs'/f'{scene}_{case}.npz',output=y)
  for small in (False,True):
   h,w=(64,96) if small else (1024,1280);ax=torch.full((1,9,h,w),.3,device='cuda');ac=torch.full((1,1,64,64),.3,device='cuda');ax,ac=prepare_inputs(m,ax,ac);path=RUN/'source_onnx'/f'{scene}_{case}{"_small" if small else ""}.onnx'
   if case!='v08_c11':torch.onnx.export(m,(ax,ac),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path)
   else:g=onnx.load(path)
   outshape=[d.dim_value for d in g.graph.output[0].type.tensor_type.shape.dim];assert outshape==[1,1,h*3,w*3],outshape
   rows.append({'file':path.name,'scene':scene,'case':case,'small':small,'input_shape':list(ax.shape),'input_dtype':str(ax.dtype),'output_shape':outshape,'ops':{op:sum(n.op_type==op for n in g.graph.node) for op in ('Shape','Gather','Concat','Reshape','Transpose','ConvTranspose')},'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_mean_gray_vs_C11':float(d.mean()),'source_max_gray_vs_C11':float(d.max()),'source_byte_identity_with_C11':bool(np.array_equal(y,baseline)),'SDK_verified':False});print('EXPORTED',scene,case,small,flush=True)
  del m;gc.collect();torch.cuda.empty_cache()
(RUN/'extra_export.json').write_text(json.dumps({'cases':rows,'complete_full_output':True,'SDK_verified':False},indent=2));print('COMPLETE',flush=True)
