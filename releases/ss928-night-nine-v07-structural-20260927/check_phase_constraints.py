"""Check native-grid equality before clipping/rounding and store weight proofs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from joint_candidates import load_joint
from phase_projection import preserve_native_means
torch.set_num_threads(2);torch.manual_seed(2749)
out=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-STRUCTURAL-20260927');rows=[]
with torch.no_grad():
 for scene in ['ordinary','special']:
  base=load_joint(scene,'baseline',out,device='cpu');original=base.core.model.upsample[0];w=original.weight.double();b=original.bias.double()
  ww,bb,proof=preserve_native_means(w.numpy(),b.numpy());_,_,stored_proof=preserve_native_means(original.weight.numpy(),original.bias.numpy())
  for kind in ['constant','random','edge_impulse']:
   x=torch.ones((1,16,8,12),dtype=torch.float64)*.3 if kind=='constant' else torch.randn((1,16,8,12),dtype=torch.float64)*.3
   if kind=='edge_impulse':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
   teacher=F.avg_pool2d(F.pixel_shuffle(F.conv2d(x,w,b,padding=1),6),3,3)
   candidate=F.avg_pool2d(F.interpolate(F.pixel_shuffle(F.conv2d(x,torch.from_numpy(ww),torch.from_numpy(bb),padding=1),3),scale_factor=2,mode='nearest'),3,3)
   error=float((teacher-candidate).abs().max());assert error<1e-12,error
   rows.append({'scene':scene,'input':kind,'max_float64_preclip_native_error':error,'linear_constraint_proof':proof,'stored_half_weight_proof':stored_proof,'SDK_verified':False})
(out/'phase_constraints.json').write_text(json.dumps({'scope':'linear outputs before clipping, rounding, or mixed-precision inference; actual quality separate','checks':rows},indent=2));print('PASSED',len(rows),flush=True)
