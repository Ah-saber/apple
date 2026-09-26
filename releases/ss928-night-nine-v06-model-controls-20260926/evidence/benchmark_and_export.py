"""Full-model controlled timing; source exports do not imply SS928 support."""
import fcntl,gc,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V06-FOLLOWUP-20260926';out.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp',str(Path(__file__).parent)];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from materialize_aliases import materialize
from load_packed_model import load_packed_model
from model_variants import CompleteVariant,phase_permutation,FixedCurrentPack
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
# A phase-index pattern verifies spatial mapping, including both ordering cases.
pattern=torch.arange(36*8*12,dtype=torch.float32).reshape(1,36,8,12)
for first,second in [(2,3),(3,2)]:
 actual=torch.nn.functional.pixel_shuffle(torch.nn.functional.pixel_shuffle(pattern[:,phase_permutation(first,second)],first),second);assert torch.equal(actual,torch.nn.functional.pixel_shuffle(pattern,6))
report={'input_dtype':'float32','output_dtype':'float32','input_shape':[1,9,1024,1280],'output_shape':[1,1,3072,3840],'GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'warmup':30,'rounds':3,'samples_per_round':200,'TF32':False,'CUDA_graphs':False,'NPU_verified':False,'spatial_phase_mapping_exact':True,'results':{}}
variants=[('base',6,1,False),('two_three',2,3,False),('three_two',3,2,False),('fixed_pack',6,1,True),('fixed_two_three',2,3,True)]
for scene,frame in [('ordinary',20),('special',60)]:
 run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={}
 for name,first,second,fixed in variants:
  base=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp);m=base if name=='base' else CompleteVariant(base,first,second,fixed)
  torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
  with torch.inference_mode():
   source=m(x,ctx);compiled=exe(x,ctx)
   if name=='base':reference=source.cpu().numpy();compiled_reference=compiled.cpu().numpy()
   source_diff=np.abs(source.cpu().numpy()-reference);compiled_diff=np.abs(compiled.cpu().numpy()-compiled_reference);numerical={'source_vs_base_max_gray':float(source_diff.max()),'source_vs_base_mean_gray':float(source_diff.mean()),'compiled_vs_base_max_gray':float(compiled_diff.max()),'compiled_vs_base_mean_gray':float(compiled_diff.mean()),'compiled_vs_source_max_gray':float((source-compiled).abs().max()),'compiled_vs_source_mean_gray':float((source-compiled).abs().mean())}
   assert numerical['source_vs_base_max_gray']<=.3,(scene,name,numerical)
   # The fixed FP32 selectors must retain the current RAW phases exactly.
   if fixed:
    f=FixedCurrentPack(base.front);picked=torch.nn.functional.conv2d(x,f.pack_weight,stride=2);expected=torch.nn.functional.pixel_unshuffle(x[:,-1:],2);assert torch.equal(picked,expected);numerical['fixed_fp32_pack_bit_exact']=True;del f,picked,expected
   del source,compiled
   for _ in range(30):exe(x,ctx)
   torch.cuda.synchronize();rounds=[]
   for _ in range(3):
    values=[]
    for _ in range(200):
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();exe(x,ctx);e.record();e.synchronize();values.append(s.elapsed_time(e))
    rounds.append(values)
   flat=sum(rounds,[]);record={'mean_ms':float(np.mean(flat)),'p95_ms':float(np.percentile(flat,95)),'samples_ms':rounds,'numerical':numerical,'first_shuffle':first,'second_shuffle':second,'fixed_pack':fixed};report['results'][scene][name]=record;print('TIMING',scene,name,{k:v for k,v in record.items() if k!='samples_ms'},flush=True)
   if name!='base':
    for dtype in ['float32','float16']:
     m.output=dtype;folder=out/'onnx'/name;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{scene}_{dtype}.onnx';tmp=folder/f'{scene}_{dtype}_raw.onnx'
     torch.onnx.export(m,(x,ctx),str(tmp),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
     original=onnx.load(str(tmp));converted,aliases=materialize(original);onnx.save(converted,str(path));shuffles=[n for n in converted.graph.node if n.op_type=='DepthToSpace'];assert len(shuffles)==(1 if second==1 else 2)
     dtype_out=onnx.TensorProto.FLOAT16 if dtype=='float16' else onnx.TensorProto.FLOAT;assert converted.graph.output[0].type.tensor_type.elem_type==dtype_out
     info={'scene':scene,'variant':name,'dtype':dtype,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'initializer_aliases':aliases,'source_model_verified':True,'vendor_R4_compatibility_applied':False,'NPU_compile_verified':False,'input_shape':[1,9,1024,1280],'output_shape':[1,1,3072,3840]};path.with_suffix('.json').write_text(json.dumps(info,indent=2));print('EXPORT',scene,name,dtype,flush=True)
  del exe,m,base;gc.collect();torch.cuda.empty_cache();(out/'gpu_and_source_verification.json').write_text(json.dumps(report,indent=2))
 del x,ctx
print('FINISHED',flush=True)
