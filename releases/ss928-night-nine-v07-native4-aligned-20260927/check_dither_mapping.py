"""Prove nominal one-ninth gray mean encoding and paired statistics algebra."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
torch.set_num_threads(2);torch.manual_seed(2797);rows=[]
x=torch.linspace(0,255,255001,dtype=torch.float64);offset=(torch.arange(9,dtype=torch.float64)-4)/9
encoded=(x[:,None]+offset).clamp(0,255).round().mean(1);error=(encoded-x).abs();assert error.max()<1/18+1e-10
rows.append({'case':'fractional_gray_encoding_float64','points':len(x),'max_gray_mean_error':float(error.max()),'nearest_integer_mean_max_error':float((x.round()-x).abs().max()),'scope':'float64 ideal offsets; stored Half error checked separately'})
for kind in ['positive','signed','edge']:
 x=torch.rand((1,9,8,12),dtype=torch.float64) if kind=='positive' else torch.randn((1,9,8,12),dtype=torch.float64)
 if kind=='edge':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
 w=torch.randn((12,9,2,2),dtype=torch.float64)*.1;plain=F.conv2d(x,w,stride=2);pair=F.relu(F.conv2d(x,torch.cat((w,-w)),torch.zeros(24,dtype=torch.float64),stride=2));restored=pair[:,:12]-pair[:,12:];err=float((restored-plain).abs().max());assert err<1e-12;rows.append({'case':'relu_statistics_pair','input':kind,'max_float64_error':err})
p=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927/dither_and_statistics_algebra.json');p.write_text(json.dumps({'controls':rows,'SDK_verified':False},indent=2));print('PASSED',len(rows),flush=True)
