"""Exact arithmetic checks for affine projection/bilinear reorder and padded input."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
torch.set_num_threads(2);torch.manual_seed(2709);rows=[]
for kind in ('constant','signed_random','edge'):
 x=torch.full((1,12,5,7),.3,dtype=torch.float64) if kind=='constant' else torch.randn(1,12,5,7,dtype=torch.float64)
 if kind=='edge':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
 p=torch.nn.Conv2d(12,16,1,dtype=torch.float64)
 a=F.interpolate(p(x),size=(19,25),mode='bilinear',align_corners=False);b=p(F.interpolate(x,size=(19,25),mode='bilinear',align_corners=False));err=float((a-b).abs().max());assert err<1e-12
 z=torch.randn(1,9,12,16,dtype=torch.float64);w=torch.randn(12,9,2,2,dtype=torch.float64);wp=F.pad(w,(0,0,0,0,0,7));zp=torch.cat((z,torch.zeros_like(z[:,:7])),1);pe=float((F.conv2d(z,w,stride=2)-F.conv2d(zp,wp,stride=2)).abs().max());assert pe<1e-12
 rows.append({'input':kind,'project_resize_max_float64':err,'padded_input_max_float64':pe})
Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927/reference_input_algebra.json').write_text(json.dumps({'controls':rows,'Half_NPU_errors_not_covered':True},indent=2));print('PASSED',rows)
