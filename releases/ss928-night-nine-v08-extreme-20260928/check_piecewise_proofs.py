import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import PiecewiseGelu
torch.set_num_threads(2);torch.manual_seed(2804);rows=[]
x=torch.linspace(-10,10,200001,dtype=torch.float64).reshape(1,1,1,-1)
for count,bound in [(13,.024),(25,.007)]:
 m=PiecewiseGelu(1,count,'cpu',torch.float64)
 with torch.no_grad():error=(m(x)-F.gelu(x)).abs();maxerror=float(error.max());index=int(error.flatten().argmax())
 assert maxerror<bound;rows.append({'knots':count,'range':[-10,10],'count':x.numel(),'numeric_max_abs_error':maxerror,'argmax_x':float(x.flatten()[index]),'asserted_bound':bound,'analytic_segment_error_bound':float(2/(2*torch.pi)**.5*(6/(count-1))**2/8+.00405),'float16_not_bit_equivalent':True})
Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927/piecewise_proofs.json').write_text(json.dumps({'checks':rows,'SDK_verified':False,'model_quality_needs_separate_sequence_test':True},indent=2));print('PIECEWISE_PROOFS',rows)
