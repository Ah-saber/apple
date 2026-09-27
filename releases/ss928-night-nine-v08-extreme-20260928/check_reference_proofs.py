"""Actual trained coefficient reference padding and affine mean reorder controls."""
import argparse,copy,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import load_deployment,PaddedReference,ReferenceMeanFirst
model=load_deployment(a.scene,'front3x3_trained_native4_alignpixel',root,device='cpu');ref=copy.deepcopy(model.core.model.global_reference).double();padded=PaddedReference(ref);meanfirst=ReferenceMeanFirst(ref);padded_mean=ReferenceMeanFirst(padded);torch.manual_seed(2801);torch.set_num_threads(2);rows=[]
def field(module,x):
 r=module.encode(x);return module.project(r+r.mean((-2,-1),keepdim=True))
with torch.inference_mode():
 for label,x in [('signed',torch.randn(1,1,64,64,dtype=torch.float64)),('constant',torch.full((1,1,64,64),.25,dtype=torch.float64)),('corner',torch.zeros(1,1,64,64,dtype=torch.float64))]:
  if label=='corner':x[...,0,0]=1
  source=field(ref,x)
  for name,module in [('pad16',padded),('meanfirst',meanfirst),('pad16_meanfirst',padded_mean)]:
   error=float((source-field(module,x)).abs().max());assert error<1e-10;rows.append({'control':label,'variant':name,'max_abs':error})
print(json.dumps({'scene':a.scene,'actual_coefficients':True,'controls':rows,'Half_rounding_not_bit_exact_asserted':True,'SDK_verified':False},indent=2))
