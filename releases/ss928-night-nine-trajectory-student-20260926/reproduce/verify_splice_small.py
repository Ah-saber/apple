"""Independent ONNX reference check of splice on a frozen source control, not real R4."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-TRAJECTORY-STUDENT-COMPACT-20260926';control=out/'splice_controls';control.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime' if (Path(__file__).parent/'runtime').is_dir() else Path(__file__).parent.parent/'runtime'),str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx.reference import ReferenceEvaluator
from load_student_model import load_student_model
from load_candidate import load_candidate
from trajectory_student import StudentByteNine
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
report={}
for scene in ('ordinary','special'):
 cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt';step=json.loads((out/f'{scene}_selection.json').read_text())['selected_step']
 m=load_student_model(out/f'{scene}_student_{step:06d}.pt',cp,bp);old=load_candidate(cp,bp,remove_zero_init=True);byte=StudentByteNine(m.core,m.student)
 x=torch.linspace(.2,.8,9*32*48,device='cuda').reshape(1,9,32,48);ctx=torch.linspace(.2,.8,64*64,device='cuda').reshape(1,1,64,64)
 source_path=control/f'{scene}_old.onnx';student_path=control/f'{scene}_student.onnx';splice_path=control/f'{scene}_spliced.onnx'
 with torch.inference_mode():
  expected=byte(x,ctx).cpu().numpy()
  torch.onnx.export(old,(x,ctx),str(source_path),input_names=['nine_raw','reference_thumb'],output_names=['display_uint8'],opset_version=17,dynamo=False)
  torch.onnx.export(m,(x,ctx),str(student_path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
 cmd=[sys.executable,str(Path(__file__).parent/'splice_r4_front.py'),'--r4',str(source_path),'--student',str(student_path),'--output',str(splice_path)]
 env=os.environ.copy();env['PYTHONPATH']='/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'
 subprocess.run(cmd,env=env,check=True,capture_output=True)
 actual=ReferenceEvaluator(str(splice_path)).run(None,{'nine_raw':x.cpu().numpy(),'reference_thumb':ctx.cpu().numpy()})[0]
 diff=np.abs(actual.astype(np.int16)-expected.astype(np.int16));assert actual.shape==expected.shape and diff.max()<=2,(actual.shape,expected.shape,diff.max())
 # Original core/output operators are protected by graph splice; no old Max chain remains.
 graph=onnx.load(str(splice_path));counts={}
 for n in graph.graph.node:counts[n.op_type]=counts.get(n.op_type,0)+1
 assert counts.get('Max',0)==0 and counts.get('Sqrt',0)==0
 report[scene]={'shape':list(actual.shape),'dtype':str(actual.dtype),'max_gray':int(diff.max()),'mean_gray':float(diff.mean()),'operators':counts,'real_R4_used':False,'verification':'Independent ONNX ReferenceEvaluator versus CUDA StudentByteNine; frozen source graph splice control'}
 print(scene,report[scene],flush=True)
(out/'splice_source_control_verification.json').write_text(json.dumps(report,indent=2))
