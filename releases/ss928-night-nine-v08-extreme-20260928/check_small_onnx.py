"""CPU ORT vs own source for small static export; full SDK remains unverified."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tag',default='small_ort');p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/data/zhangbenzhuang/huawei_sr/runs/TASK-019-export-dependencies/site-packages','/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnxruntime as ort
from deployment_candidates import load_deployment,prepare_inputs
torch.set_num_threads(2);torch.manual_seed(2805);rows=[];options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
for case in a.cases:
 path=r/'source_onnx'/f'{a.scene}_{case}_small.onnx';m=load_deployment(a.scene,case,r,device='cpu');x=(torch.rand(1,9,64,96)-.25).float();c=torch.rand(1,1,64,64);x,c=prepare_inputs(m,x,c)
 with torch.inference_mode():y=m(x,c).float().numpy()
 session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']);actual=session.run(None,{'nine_raw':x.numpy(),'reference_thumb':c.numpy()})[0].astype(np.float32);d=np.abs(y-actual);assert np.isfinite(actual).all();assert actual.shape==y.shape;assert float(d.mean())<=.5 and float(np.percentile(d,95))<=2.0;rows.append({'case':case,'graph_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(y.shape),'mean_gray':float(d.mean()),'p95_gray':float(np.percentile(d,95)),'max_gray':float(d.max()),'dtype':str(actual.dtype),'CPU_ORT_only':True,'SDK_verified':False});print('SMALL_ORT',rows[-1],flush=True)
(r/f'{a.scene}_{a.tag}.json').write_text(json.dumps({'seed':2805,'same_source_floating_point_execution_difference_no_bit_identity_claim':True,'gate_mean_gray':.5,'gate_p95_gray':2.0,'results':rows},indent=2))
