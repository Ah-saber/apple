"""Validate head-boundary splice on old source graph, independent CPU ONNX evaluator."""
import fcntl,json,os,sys,subprocess
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926';ctl=out/'splice_controls';ctl.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx.reference import ReferenceEvaluator
from load_packed_model import load_packed_model
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=torch.linspace(.2,.8,9*32*48,device='cuda').reshape(1,9,32,48);ctx=torch.linspace(.2,.8,64*64,device='cuda').reshape(1,1,64,64);report={}
for scene in ('ordinary','special'):
 cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 for variant in ('joint12','joint8'):
  run=out if variant=='joint12' else root/'runs/SS928-PACKED-FRONT-COMPACT8-20260926';step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];m=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp)
  src=ctl/f'{scene}_{variant}.onnx';dst=ctl/f'{scene}_{variant}_spliced.onnx';old=root/'runs/SS928-TRAJECTORY-STUDENT-COMPACT-20260926/splice_controls'/f'{scene}_old.onnx'
  with torch.inference_mode():
   expect=m(x,ctx).round().to(torch.uint8).cpu().numpy()
   torch.onnx.export(m,(x,ctx),str(src),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
  env=os.environ.copy();env['PYTHONPATH']='/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'
  subprocess.run([sys.executable,str(Path(__file__).parent/'splice_r4_packed.py'),'--r4',str(old),'--student',str(src),'--output',str(dst)],env=env,check=True,capture_output=True)
  actual=ReferenceEvaluator(str(dst)).run(None,{'nine_raw':x.cpu().numpy(),'reference_thumb':ctx.cpu().numpy()})[0];d=np.abs(actual.astype(np.int16)-expect.astype(np.int16));assert d.max()<=2,(scene,variant,d.max())
  graph=onnx.load(str(dst));counts={}
  for n in graph.graph.node:counts[n.op_type]=counts.get(n.op_type,0)+1
  assert not counts.get('Max') and not counts.get('Sqrt');assert tuple(actual.shape)==(1,1,96,144)
  item={'shape':list(actual.shape),'output_dtype':str(actual.dtype),'max_gray':int(d.max()),'mean_gray':float(d.mean()),'operators':counts,'real_R4_used':False,'NPU_verified':False,'verification':'Independent CPU ONNX evaluator versus CUDA full packed model; original source graph frozen core/output preserved'};report[f'{scene}_{variant}']=item;print(scene,variant,item,flush=True)
(out/'packed_splice_verification.json').write_text(json.dumps(report,indent=2))
