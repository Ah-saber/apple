"""Export full nine-frame/full-gray models and reference vectors for board compilation."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--compact',action='store_true');args=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs'/('SS928-TRAJECTORY-STUDENT-COMPACT-20260926' if args.compact else 'SS928-TRAJECTORY-STUDENT-20260926');(out/'onnx').mkdir(exist_ok=True);(out/'reference_outputs').mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime' if (Path(__file__).parent/'runtime').is_dir() else Path(__file__).parent.parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from load_student_model import load_student_model
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
results={}
for scene,frame in [('ordinary',20),('special',60)]:
 selection=json.loads((out/f'{scene}_selection.json').read_text());sp=out/f'{scene}_student_{selection["selected_step"]:06d}.pt'
 cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 model=load_student_model(sp,cp,bp)
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:
  x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 with torch.inference_mode():
  value=model(x,ctx);assert tuple(value.shape)==(1,1,3072,3840) and torch.isfinite(value).all()
  path=out/'onnx'/f'{scene}_trajectory_student_float32.onnx'
  torch.onnx.export(model,(x,ctx),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,do_constant_folding=True,dynamo=False)
  np.savez_compressed(out/'reference_outputs'/f'{scene}_frame_{frame}_student.npz',display_gray=value.cpu().numpy(),display_uint8=value.round().to(torch.uint8).cpu().numpy())
 checked=onnx.load(str(path));onnx.checker.check_model(checked);counts={}
 for node in checked.graph.node:counts[node.op_type]=counts.get(node.op_type,0)+1
 assert counts.get('Sqrt',0)==0 and counts.get('Max',0)==0 and counts.get('ConvTranspose',0)==1
 record={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'student_checkpoint':str(sp),'output_shape':[1,1,3072,3840],'output_dtype':'float32','input_dtype':'float32','operators':counts,'onnx_checker_passed':True,'r4_vendor_rewrites_applied':False,'npu_compile_verified':False,'npu_timing_verified':False}
 (out/'onnx'/f'{scene}_trajectory_student_float32.json').write_text(json.dumps(record,indent=2));results[scene]=record;print('EXPORT',scene,record,flush=True)
 del model,value,x,ctx;torch.cuda.empty_cache()
(out/'export.json').write_text(json.dumps(results,indent=2))
