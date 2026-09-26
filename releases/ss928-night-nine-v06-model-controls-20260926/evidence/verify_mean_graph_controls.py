"""Compare graph rewriting against an independently exported source candidate."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V06-FOLLOWUP-20260926';ctl=out/'graph_controls'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp',str(Path(__file__).parent)];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx.reference import ReferenceEvaluator
from load_packed_model import load_packed_model
from model_variants import MeanRawDitherDisplay
from rewrite_mean_raw import rewrite_mean
from rewrite_native_r2 import rewrite
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=torch.linspace(.2,.8,9*32*48,device='cuda').reshape(1,9,32,48);ctx=torch.linspace(.2,.8,64*64,device='cuda').reshape(1,1,64,64);feed={'nine_raw':x.cpu().numpy(),'reference_thumb':ctx.cpu().numpy()};report={}
for scene in ['ordinary','special']:
 run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 m=MeanRawDitherDisplay(load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp));src=ctl/f'{scene}_dither_source.onnx'
 with torch.inference_mode():torch.onnx.export(m,(x,ctx),str(src),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
 expected=ReferenceEvaluator(str(src)).run(None,feed)[0];original=onnx.load(str(ctl/f'{scene}_original.onnx'));cases={}
 for name,two,fixed in [('one',False,False),('two',True,False),('fixed',False,True),('dcr_one',False,False)]:
  old=rewrite(original,'six',True)[0] if fixed else original
  if name=='dcr_one':
   old=onnx.ModelProto();old.CopyFrom(original);next(a for n in old.graph.node if n.op_type=='DepthToSpace' for a in n.attribute if a.name=='mode').s=b'DCR'
  model,record=rewrite_mean(old,two);onnx.save(model,str(ctl/f'{scene}_mean_{name}.onnx'));actual=ReferenceEvaluator(model).run(None,feed)[0];delta=np.abs(actual-expected)
  assert delta.max()<=.001,(scene,name,float(delta.max()))
  cases[name]={'max_gray':float(delta.max()),'mean_gray':float(delta.mean()),'bit_exact':bool(np.array_equal(actual,expected)),'shape':list(actual.shape),'dtype':str(actual.dtype),'original_IO_preserved':model.graph.input==original.graph.input and model.graph.output==original.graph.output,'real_native_r2_used':False,'NPU_verified':False};print(scene,name,cases[name],flush=True)
 # Guards reject graphs whose phase order, clipping, or side branches differ.
 rejected={}
 for bad in ['phase_order','clip_bounds','extra_shape_consumer']:
  broken=onnx.ModelProto();broken.CopyFrom(original)
  if bad=='phase_order':
   n=next(n for n in broken.graph.node if n.op_type=='DepthToSpace');next(a for a in n.attribute if a.name=='mode').s=b'invalid'
  elif bad=='clip_bounds':
   n=next(n for n in broken.graph.node if n.op_type=='Clip');n.input[2]=n.input[1]
  else:
   n=next(n for n in reversed(broken.graph.node) if n.op_type=='Conv');broken.graph.node.append(onnx.helper.make_node('Shape',[n.output[0]],['guard_observed_shape']));broken.graph.output.append(onnx.helper.make_tensor_value_info('guard_observed_shape',onnx.TensorProto.INT64,[4]))
  try:rewrite_mean(broken)
  except ValueError as e:rejected[bad]=str(e)
  else:raise AssertionError('Unsafe graph accepted: '+bad)
 report[scene]={'cases':cases,'guard_rejections':rejected,'independent_reference':'ONNX ReferenceEvaluator exported candidate vs rewritten original'}
(out/'mean_graph_control_verification.json').write_text(json.dumps(report,indent=2))
