"""Check full-image float64 fusion, including original zero-padding boundary."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from continuation_candidates import TailFusedOutput
torch.set_num_threads(2);torch.manual_seed(2803);rows=[]
for kind in ['constant','random','edge']:
 x=torch.full((1,16,8,12),.3,dtype=torch.float64) if kind=='constant' else torch.randn((1,16,8,12),dtype=torch.float64)*.3
 if kind=='edge':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
 tail=torch.nn.Conv2d(16,16,3,padding=1,dtype=torch.float64);projection=torch.nn.Conv2d(16,9,3,padding=1,dtype=torch.float64);target=projection(tail(x))
 exact=TailFusedOutput(tail,projection,True).phases(x);approx=TailFusedOutput(tail,projection,False).phases(x);err=float((exact-target).abs().max());interior=float((approx-target)[:,:,1:-1,1:-1].abs().max());assert err<1e-12 and interior<1e-12
 rows.append({'input':kind,'full_exact_edge_max_float64_error':err,'interior_fusion_max_float64_error':interior,'approx_full_error_without_edge_correction':float((approx-target).abs().max())})
out=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927/tail_fusion_algebra.json');out.write_text(json.dumps({'controls':rows,'SDK_verified':False,'Half_quantization_errors_separate':True},indent=2));print('PASSED',len(rows),flush=True)
