"""Even-sensor exact signed statistics2 -> padded statistics3 controls."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
torch.set_num_threads(2);torch.manual_seed(2709);rows=[]
for h,w in ((2,2),(12,16),(64,96)):
 for kind in ('constant','signed_random','edge'):
  x=torch.full((1,9,h,w),.3,dtype=torch.float64) if kind=='constant' else torch.randn(1,9,h,w,dtype=torch.float64)
  if kind=='edge':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
  k=torch.randn(12,9,2,2,dtype=torch.float64);target=F.conv2d(x,k,stride=2);k3=torch.zeros(12,9,3,3,dtype=torch.float64);k3[:,:,1:,1:]=k;p=torch.zeros(32,9,3,3,dtype=torch.float64);p[:12]=k3;p[16:28]=-k3;v=F.relu(F.conv2d(x,p,stride=2,padding=1));rectified=v[:,:12]-v[:,16:28];k16=F.pad(k3,(0,0,0,0,0,7));x16=torch.cat((x,torch.zeros_like(x[:,:7])),1)
  errors={'linear':float((F.conv2d(x,k3,stride=2,padding=1)-target).abs().max()),'rectified32':float((rectified-target).abs().max()),'input16':float((F.conv2d(x16,k16,stride=2,padding=1)-target).abs().max())};assert max(errors.values())<1e-12;rows.append({'input':kind,'sensor_shape':[h,w],'max_float64_error':errors})
Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927/stats3_algebra.json').write_text(json.dumps({'controls':rows,'requires_even_sensor_dimensions':True,'Half_and_SDK_not_covered':True},indent=2));print('PASSED',len(rows))
