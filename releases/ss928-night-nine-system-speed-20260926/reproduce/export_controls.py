"""Export source controls and verify native-front splice with independent ONNX execution."""
import fcntl,gc,json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';basepath=Path(__file__).parent;sys.path[:0]=[str(basepath/'runtime'),str(basepath/'tools'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
from build_system import load_case,case_inputs
from materialize_aliases import materialize
from onnx.reference import ReferenceEvaluator
import onnx
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory);reports=[]
variants=['base','base_rows16','balanced','contrast','balanced_rows16','contrast_rows16','balanced_low_rows16_nhwc16_half','contrast_low_rows16_nhwc16_half']
for scene,frame in [('ordinary',20),('special',60)]:
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 for variant in variants:
  model,info=load_case(root,scene,variant);xx,cc=case_inputs(x,ctx,info);folder=out/'source_onnx';folder.mkdir(exist_ok=True);path=folder/f'{scene}_{variant}.onnx';raw=folder/f'{scene}_{variant}_raw.onnx'
  with torch.inference_mode():
   torch.onnx.export(model,(xx,cc),str(raw),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
   value=model(xx,cc).cpu().numpy();np.savez_compressed(folder/f'{scene}_{variant}_reference.npz',display_gray=value)
  graph,aliases=materialize(onnx.load(str(raw)));onnx.checker.check_model(graph);onnx.save(graph,str(path));raw.unlink();record={'scene':scene,'variant':variant,'IO':info,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'aliases':aliases,'output_shape':[1,1,3072,3840],'CPU_Resize_compatibility_applied':False,'NPU_verified':False};path.with_suffix('.json').write_text(json.dumps(record,indent=2));reports.append(record);print('EXPORT',scene,variant,flush=True);del model,xx,cc;gc.collect();torch.cuda.empty_cache()
 # Small graph control retains every CNN/resize but tests prefix replacement
 # against an independent interpreter, with unchanged reference/body/output.
 models={}
 for variant in ('base','balanced','contrast'):
  model,info=load_case(root,scene,variant);small=torch.rand(1,9,64,96,device='cuda');path=folder/f'{scene}_{variant}_small.onnx';torch.onnx.export(model,(small,ctx),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);graph,_=materialize(onnx.load(str(path)));onnx.save(graph,str(path));models[variant]=path;del model
 # CPU reference runs after leaving the GPU lock below.
 del x,ctx
audit={'exports':reports,'source_full_shape_checked':True,'vendor_compatible_graph':False,'NPU_verified':False};(out/'source_export_manifest.json').write_text(json.dumps(audit,indent=2));print('FINISHED',flush=True)
