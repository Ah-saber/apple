"""Independent CPU execution of graph rewrites; source controls, not native_r2."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V06-FOLLOWUP-20260926';ctl=out/'graph_controls';ctl.mkdir(exist_ok=True)
sys.path[:0]=['/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp',str(Path(__file__).parent)];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx.reference import ReferenceEvaluator
from rewrite_native_r2 import rewrite
from materialize_aliases import materialize
from load_packed_model import load_packed_model
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=torch.linspace(.2,.8,9*32*48,device='cuda').reshape(1,9,32,48);ctx=torch.linspace(.2,.8,64*64,device='cuda').reshape(1,1,64,64);feed={'nine_raw':x.cpu().numpy(),'reference_thumb':ctx.cpu().numpy()};report={}
for scene in ['ordinary','special']:
 run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 m=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp);src=ctl/f'{scene}_original.onnx'
 with torch.inference_mode():torch.onnx.export(m,(x,ctx),str(src),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
 original=onnx.load(str(src));expected=ReferenceEvaluator(original).run(None,feed)[0];mat,aliases=materialize(original);array=ReferenceEvaluator(mat).run(None,feed)[0];assert np.array_equal(expected,array)
 cases={}
 for name,order,fixed in [('two_three','two_three',False),('three_two','three_two',False),('fixed_pack','six',True),('fixed_two_three','two_three',True)]:
  model,record=rewrite(original,order,fixed);path=ctl/f'{scene}_{name}.onnx';onnx.save(model,str(path));actual=ReferenceEvaluator(model).run(None,feed)[0];delta=np.abs(actual.astype(np.float32)-expected.astype(np.float32));assert delta.max()<=.125,(scene,name,float(delta.max()))
  cases[name]={'max_gray':float(delta.max()),'mean_gray':float(delta.mean()),'bit_exact':bool(np.array_equal(actual,expected)),'shape':list(actual.shape),'dtype':str(actual.dtype),'original_IO_preserved':model.graph.input==original.graph.input and model.graph.output==original.graph.output,'real_native_r2_used':False,'NPU_verified':False};print(scene,name,cases[name],flush=True)
 report[scene]={'alias_materialization_bit_exact':True,'alias_count':len(aliases),'cases':cases}
(out/'graph_control_verification.json').write_text(json.dumps(report,indent=2))
