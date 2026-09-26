"""Independent CPU ONNX controls; real R4 is unavailable."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
from onnx.reference import ReferenceEvaluator
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926';ctl=out/'output_controls';ctl.mkdir(exist_ok=True)
x=np.linspace(.2,.8,9*32*48,dtype=np.float32).reshape(1,9,32,48);ctx=np.linspace(.2,.8,64*64,dtype=np.float32).reshape(1,1,64,64)
report={};env=os.environ.copy();env['PYTHONPATH']='/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'
for scene in ('ordinary','special'):
 src=root/'runs/SS928-TRAJECTORY-STUDENT-COMPACT-20260926/splice_controls'/f'{scene}_student.onnx';base=ReferenceEvaluator(str(src)).run(None,{'nine_raw':x,'reference_thumb':ctx})[0]
 for dtype,dcr in [('float32',False),('float32',True),('float16',True)]:
  dst=ctl/f'{scene}_{dtype}_{dcr}.onnx';cmd=[sys.executable,str(Path(__file__).parent/'rewrite_r4_output.py'),'--input',str(src),'--output',str(dst),'--dtype',dtype]
  if dcr:cmd.append('--dcr')
  subprocess.run(cmd,env=env,check=True,capture_output=True)
  actual=ReferenceEvaluator(str(dst)).run(None,{'nine_raw':x,'reference_thumb':ctx})[0];delta=np.abs(actual.astype(np.float32)-base)
  if dtype=='float32':assert np.array_equal(actual,base),delta.max()
  else:assert np.array_equal(actual,base.astype(np.float16)),delta.max()
  report[f'{scene}_{dtype}_{dcr}']={'max_gray':float(delta.max()),'mean_gray':float(delta.mean()),'shape':list(actual.shape),'dtype':str(actual.dtype),'float32_exact':dtype=='float32','float16_matches_final_rounding':dtype=='float16','real_R4_used':False,'NPU_verified':False}
  print(scene,dtype,dcr,report[f'{scene}_{dtype}_{dcr}'],flush=True)
(out/'output_rewrite_verification.json').write_text(json.dumps(report,indent=2))
