"""Paired compiled full-model tests and source exports for RAW gray head controls."""
import fcntl,gc,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V06-FOLLOWUP-20260926'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp',str(Path(__file__).parent)];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from load_packed_model import load_packed_model
from model_variants import CompleteVariant,MeanRawDitherDisplay
from materialize_aliases import materialize
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
report={'input_dtype':'float32','output_dtype':'float32','GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'warmup':30,'rounds':3,'samples_per_round':200,'TF32':False,'CUDA_graphs':False,'NPU_verified':False,'results':{}}
for scene,frame in [('ordinary',20),('special',60)]:
 run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={}
 for name in ['base','dither_one','dither_two','dither_fixed']:
  base=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp)
  if name=='base':m=base
  else:
   if name=='dither_fixed':base=CompleteVariant(base,fixed_pack=True)
   m=MeanRawDitherDisplay(base,two_stage=name=='dither_two')
  torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
  with torch.inference_mode():
   source=m(x,ctx);value=exe(x,ctx);num={'compiled_vs_source_max_gray':float((source-value).abs().max()),'compiled_vs_source_mean_gray':float((source-value).abs().mean())}
   if name=='dither_one':reference=source.cpu().numpy()
   elif name!='base':
    delta=np.abs(source.cpu().numpy()-reference);num['source_vs_dither_one_max_gray']=float(delta.max());num['source_vs_dither_one_mean_gray']=float(delta.mean());assert delta.max()<=.0001,(name,float(delta.max()))
   del source,value
   for _ in range(30):exe(x,ctx)
   torch.cuda.synchronize();rounds=[]
   for _ in range(3):
    values=[]
    for _ in range(200):
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();exe(x,ctx);e.record();e.synchronize();values.append(s.elapsed_time(e))
    rounds.append(values)
   flat=sum(rounds,[]);record={'mean_ms':float(np.mean(flat)),'p95_ms':float(np.percentile(flat,95)),'samples_ms':rounds,'numerical':num};report['results'][scene][name]=record;print('TIMING',scene,name,{k:v for k,v in record.items() if k!='samples_ms'},flush=True)
   if name!='base':
    folder=out/'onnx'/name;folder.mkdir(parents=True,exist_ok=True);refdir=out/'reference_outputs'/name;refdir.mkdir(parents=True,exist_ok=True)
    for dtype in ['float32','float16']:
     m.output=dtype;value=m(x,ctx);path=folder/f'{scene}_{dtype}.onnx';tmp=folder/f'{scene}_{dtype}_raw.onnx'
     torch.onnx.export(m,(x,ctx),str(tmp),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
     checked,aliases=materialize(onnx.load(str(tmp)));onnx.save(checked,str(path));counts={}
     for n in checked.graph.node:counts[n.op_type]=counts.get(n.op_type,0)+1
     assert tuple(value.shape)==(1,1,3072,3840) and counts.get('ConvTranspose')==2 and counts.get('DepthToSpace',0)==(1 if name=='dither_two' else 0)
     data=value.cpu().numpy();info={'scene':scene,'variant':name,'output_dtype':dtype,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'operators':counts,'alias_count':len(aliases),'input_shape':[1,9,1024,1280],'output_shape':[1,1,3072,3840],'constant_phase_offsets_gray':[-4/9,4/9],'learning_weights':'averaged last output phases; no retraining','NPU_compile_verified':False,'R2_vendor_compatibility_applied':False}
     path.with_suffix('.json').write_text(json.dumps(info,indent=2))
     if dtype=='float32':np.savez_compressed(refdir/f'{scene}_frame_{frame}.npz',display_gray=data,display_uint8=np.rint(data).astype(np.uint8))
     print('EXPORT',scene,name,dtype,counts,flush=True);del value,data
  del exe,m,base;gc.collect();torch.cuda.empty_cache();(out/'mean_raw_gpu_timing.json').write_text(json.dumps(report,indent=2))
 del x,ctx
print('FINISHED',flush=True)
