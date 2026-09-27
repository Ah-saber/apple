import sys,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import DitherPhaseOutput
torch.set_num_threads(2);torch.manual_seed(2806);rows=[]
for factor in (2,4):
 for h,w in [(2,3),(8,12)]:
  old=nn.Module();old.conv=nn.Conv2d(16,16,3,padding=1).double();x=torch.randn(1,16,h,w,dtype=torch.float64)
  with torch.no_grad():
   phases=old.conv(x).clamp(0,1)*255;raw=F.pixel_shuffle(phases[:,:factor**2],factor);tile=(torch.arange(9,dtype=torch.float64).reshape(1,1,3,3)-4)/9;expected=(F.interpolate(raw,scale_factor=3,mode='nearest')+tile.repeat(1,1,raw.shape[2],raw.shape[3])).clamp(0,255);actual=DitherPhaseOutput(old,factor)(x);error=float((actual-expected).abs().max())
  assert error<1e-10;rows.append({'factor':factor,'shape':[h,w],'double_max_abs':error})
Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927/dither_proofs.json').write_text(json.dumps({'seed':2806,'checks':rows,'offset_range_gray':[-4/9,4/9],'static_pattern_not_real_3x_detail':True,'display_float16_not_bit_equivalent':True,'SDK_verified':False},indent=2));print('DITHER_PROOFS',len(rows))
