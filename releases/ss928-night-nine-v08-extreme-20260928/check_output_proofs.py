"""Double precision full-border layout and native four-phase mean controls."""
import json,sys,copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from extreme_candidates import SplitOutput,NativeOutput,PlainFront
class Original(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Conv2d(16,16,3,padding=1).double()
 def forward(self,x):
  return F.interpolate(F.pixel_shuffle((self.conv(x)[:,:9].clamp(0,1)*255),3),scale_factor=2,mode='nearest')
torch.manual_seed(2794);torch.set_num_threads(2);rows=[]
for h,w in ((3,4),(8,11),(12,14)):
 original=Original();split=SplitOutput(original);native=NativeOutput(original)
 with torch.no_grad():
  for label,x in [('signed',torch.randn(1,16,h,w,dtype=torch.float64)),('constant',torch.ones(1,16,h,w,dtype=torch.float64)),('corner',torch.zeros(1,16,h,w,dtype=torch.float64))]:
   if label=='corner':x[...,0,0]=1
   difference=float((original(x)-split(x)).abs().max());assert difference<1e-10
   unclipped=F.interpolate(F.pixel_shuffle(original.conv(x)[:,:9],3),scale_factor=2,mode='nearest');raw=F.avg_pool2d(unclipped,3,3);expected=F.pixel_shuffle(native.conv(x),2);mean_difference=float((raw-expected).abs().max());assert mean_difference<1e-12
   rows.append({'shape':[h,w],'control':label,'split3_after_clip_double_max_abs':difference,'unclipped_native4_mean_double_max_abs':mean_difference})
front=PlainFront(3,1).double()
for h,w in ((8,10),(12,14)):
 x=torch.randn(1,9,h,w,dtype=torch.float32).double();y=front(x);expected=F.pixel_unshuffle(x[:,-1:],2);assert torch.equal(y,expected)
print(json.dumps({'seed':2794,'controls':rows,'plain_front_init_signed_current_exact':True,'native4_clipping_rounding_or_triple_detail_preserved':False,'SDK_verified':False},indent=2))
