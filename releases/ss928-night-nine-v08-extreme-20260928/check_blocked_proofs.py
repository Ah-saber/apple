"""Exact all-pixel low-resolution row grouping, full unpack retained in graph."""
import json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from deployment_candidates import BlockedOutput
class Projection(nn.Module):
 def __init__(self):super().__init__();self.conv=nn.Conv2d(16,16,3,padding=1).double()
torch.manual_seed(2802);torch.set_num_threads(2);rows=[]
with torch.inference_mode():
 for factor in (2,4):
  group=16//factor
  for h,w in ((group,5),(group*2,7),(group*3,9)):
   old=Projection();x=torch.randn(1,16,h,w,dtype=torch.float64);p=old.conv(x)[:,:factor**2].clamp(0,1)*255;expected=F.interpolate(F.pixel_shuffle(p,factor),scale_factor=3,mode='nearest')
   for split in (False,True):
    model=BlockedOutput(old,factor,full=True,split=split);error=float((expected-model(x)).abs().max());assert error<1e-10;rows.append({'factor':factor,'shape':[h,w],'split':split,'all_pixels_double_max_abs':error})
print(json.dumps({'controls':rows,'full_gray_restore_inside_graph':True,'blocked_only_diagnostic_not_goal':True,'SDK_verified':False},indent=2))
