"""Independent float64 algebra checks of phase means, fusion, and scale movement."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from continuation_candidates import FusedNineOutput,ScaledPhaseOutput,NativeFourOutput
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927';torch.set_num_threads(2);torch.manual_seed(2791);rows=[]
for kind in ['constant','random','edge']:
 x=torch.ones((1,16,3,5),dtype=torch.float64)*.3 if kind=='constant' else torch.randn((1,16,3,5),dtype=torch.float64)*.3
 if kind=='edge':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
 projection=torch.nn.Conv2d(16,9,3,padding=1,dtype=torch.float64);projection.weight.data.mul_(.1);projection.bias.data.mul_(.1)
 old=F.interpolate(F.pixel_shuffle(projection(x),3),scale_factor=2,mode='nearest')
 for factor,mode in [(3,'channel'),(3,'image'),(6,'channel')]:
  y=FusedNineOutput(projection,factor,mode)(x);err=float((y-old).abs().max());assert err<1e-12;rows.append({'input':kind,'case':f'fused9_{factor}_{mode}','max_float64_error':err})
 for mode in ['exact','folded']:
  y=ScaledPhaseOutput(projection,3,mode)(x);err=float((y.double()-old.clamp(0,1)*255).abs().max());assert err<1e-5;rows.append({'input':kind,'case':'scaled9_'+mode,'max_output_cast_float32_error_gray':err})
 # Native4 init must equal pre-clip original phase means, not clipped means.
 projection36=torch.nn.Conv2d(16,36,3,padding=1,dtype=torch.float64);native=NativeFourOutput(projection36,'linear');native.first.double()
 y=native.phases(x);target=projection36(x).reshape(1,2,3,2,3,3,5).mean((2,4)).reshape(1,4,3,5)
 err=float((y-target).abs().max());assert err<1e-6;rows.append({'input':kind,'case':'native4_linear_preclip','max_error':err,'weights_initialized_float32':True})
(out/'algebra_checks.json').write_text(json.dumps({'controls':rows,'NPU_verified':False,'finite_precision_source_quality_separate':True},indent=2));print('PASSED',len(rows),flush=True)
