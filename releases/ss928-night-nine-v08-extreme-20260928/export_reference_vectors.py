"""Source-matched full outputs for actual deployment input vectors, no GT."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tag',default='manifest');p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');r=root/'runs/SS928-EXTREME-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
l=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
frame=20 if a.scene=='ordinary' else 60;source=root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{a.scene}_frame_{frame}.npz';v=np.load(source);x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda();out=r/'reference_outputs';out.mkdir(exist_ok=True);inp=r/'test_vectors';inp.mkdir(exist_ok=True);np.savez_compressed(inp/source.name,nine_raw=v['nine_raw'],reference_thumb=v['reference_thumb']);rows=[]
with torch.inference_mode():
 for case in a.cases:
  m=load_deployment(a.scene,case,r);xx,cc=prepare_inputs(m,x,c);y=m(xx,cc).cpu().numpy();file=out/f'{a.scene}_{case}.npz'
  if file.exists():raise FileExistsError(file)
  assert np.isfinite(y).all();np.savez_compressed(file,output=y);inputs=[]
  for name,t in [('nine_raw',xx),('reference_thumb',cc)]:
   z=t.cpu().numpy();inputs.append({'name':name,'shape':list(z.shape),'dtype':str(z.dtype),'sha256':hashlib.sha256(z.tobytes()).hexdigest(),'bytes':z.nbytes})
  rows.append({'case':case,'file':file.name,'frame':frame,'source_input_file':source.name,'input_file_sha256':hashlib.sha256((inp/source.name).read_bytes()).hexdigest(),'graph_file':f'{a.scene}_{case}.onnx','output_shape':list(y.shape),'output_dtype':str(y.dtype),'output_sha256_compact':hashlib.sha256(y.tobytes()).hexdigest(),'npz_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'inputs':inputs,'source_execution':True,'SDK_verified':False,'GT_used':False});print('REFERENCE',a.scene,case,flush=True);del m,xx,cc,y;torch.cuda.empty_cache()
(out/f'{a.scene}_{a.tag}.json').write_text(json.dumps({'scene':a.scene,'frame':frame,'models':rows,'nine_actual_frames':True,'SDK_output_must_match_declared_dtype_and_compact_layout':True},indent=2))
