"""Static complete models; no SS928 vendor rewrite or board validation implied."""
import fcntl,hashlib,json,sys,gc
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/trajectory_student_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from load_packed_model import load_packed_model
from load_student_model import load_student_model
from output_variant import HalfShuffleStudent
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
report={}
for scene,frame in [('ordinary',20),('special',60)]:
 cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:
  x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 for variant in ('joint12','joint8','s8_halfshuffle'):
  dest=out/variant;(dest/'onnx').mkdir(parents=True,exist_ok=True);(dest/'reference_outputs').mkdir(exist_ok=True)
  for dtype in ('float32','float16'):
   if variant=='s8_halfshuffle':
    run=root/'runs/SS928-TRAJECTORY-STUDENT-COMPACT-20260926';step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];ck=run/f'{scene}_student_{step:06d}.pt';loaded=load_student_model(ck,cp,bp);m=HalfShuffleStudent(loaded.core,loaded.student,output=dtype)
   else:
    run=root/'runs'/('SS928-PACKED-FRONT-20260926' if variant=='joint12' else 'SS928-PACKED-FRONT-COMPACT8-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];ck=run/f'{scene}_packed_front_{step:06d}.pt';m=load_packed_model(ck,cp,bp,output=dtype)
   with torch.inference_mode():
    value=m(x,ctx);assert tuple(value.shape)==(1,1,3072,3840) and torch.isfinite(value).all();arr=value.cpu().numpy()
    path=dest/'onnx'/f'{scene}_{dtype}.onnx'
    torch.onnx.export(m,(x,ctx),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,do_constant_folding=True,dynamo=False)
   graph=onnx.load(str(path));onnx.checker.check_model(graph);counts={}
   for n in graph.graph.node:counts[n.op_type]=counts.get(n.op_type,0)+1
   assert counts.get('Max',0)==0 and counts.get('Sqrt',0)==0 and counts.get('ConvTranspose',0)==1
   if dtype=='float32':
    before=arr;refpath=dest/'reference_outputs'/f'{scene}_frame_{frame}.npz';np.savez_compressed(refpath,display_gray=arr,display_uint8=np.rint(arr).astype(np.uint8))
    if variant=='s8_halfshuffle':
     old=loaded(x,ctx).detach().cpu().numpy();assert np.array_equal(old,arr),np.abs(old-arr).max()
   else:
    assert np.array_equal(arr,before.astype(np.float16)),np.abs(arr-before.astype(np.float16)).max()
   record={'variant':variant,'scene':scene,'step':step,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_dtype':'float32','output_dtype':dtype,'full_input_shape':[1,9,1024,1280],'full_output_shape':[1,1,3072,3840],'operators':counts,'onnx_checker_passed':True,'NPU_compile_verified':False,'NPU_timing_verified':False,'half_shuffle_float32_source_equivalent':variant=='s8_halfshuffle','float16_matches_float32_cast':dtype=='float16','float16_max_rounding_gray':float(np.abs(before-arr.astype(np.float32)).max()) if dtype=='float16' else 0.}
   path.with_suffix('.json').write_text(json.dumps(record,indent=2));report[f'{scene}_{variant}_{dtype}']=record;print('EXPORT',scene,variant,dtype,record,flush=True)
   del m,value,arr;gc.collect();torch.cuda.empty_cache()
 del x,ctx
(out/'export.json').write_text(json.dumps(report,indent=2))
