"""Decode layout and zero-slot statistics controls, using actual fixed weights."""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from body_candidates import load_body
torch.set_num_threads(2);torch.manual_seed(2791);root=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BODY-DISTILL-20260927');rows=[]
for scene in ('ordinary','special'):
    system=load_body(scene,'combo_stats3_relu32_aligned16_half_output_fixed',root,device='cpu');weight=system.front.weight.double();pad_weight=torch.cat((weight,torch.zeros_like(weight[:,:7])),1)
    for kind in ('signed_random','constant','corner_impulse'):
        x=torch.randn(1,9,24,26,dtype=torch.float64) if kind=='signed_random' else torch.full((1,9,24,26),.3,dtype=torch.float64)
        if kind=='corner_impulse':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-1
        padded=torch.cat((x,torch.zeros_like(x[:,:7])),1);nhwc=padded.permute(0,2,3,1).contiguous();native=nhwc[:,None]
        assert torch.equal(nhwc.permute(0,3,1,2),padded) and torch.equal(native.reshape(1,24,26,16).permute(0,3,1,2),padded)
        expected=F.conv2d(x,weight,stride=2,padding=1);actual=F.conv2d(padded,pad_weight,stride=2,padding=1);error=float((actual-expected).abs().max());assert error<1e-12
        rows.append({'scene':scene,'input':kind,'decoded_data_identical':True,'stats_float64_max_abs':error})
print(json.dumps({'actual_fixed_coefficients':True,'controls':rows,'SDK_rank_layout_support_verified':False},indent=2))
