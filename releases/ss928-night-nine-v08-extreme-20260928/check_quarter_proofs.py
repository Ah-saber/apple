"""Signed phase preservation and decomposed output full-border controls."""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from quarter_candidates import QuarterFront,QuarterOutput
rows=[];torch.manual_seed(2800);torch.set_num_threads(2)
for kind in ('k4','3x3'):
 for width in (16,32):
  front=QuarterFront(kind,width).double()
  for h,w in ((8,12),(16,20),(24,28)):
   x=torch.randn(1,9,h,w,dtype=torch.float64);v=front(x);raw=F.pixel_unshuffle(x[:,-1:],4);expected=raw+.5 if width==16 else raw
   actual=v if width==16 else v[:,:16]-v[:,16:];error=float((actual-expected).abs().max());assert error<1e-12
   rows.append({'kind':kind,'width':width,'raw_shape':[h,w],'signed_current_phase_init_max_abs':error})
out=QuarterOutput('deconv').double();x=torch.randn(1,16,5,7,dtype=torch.float64)
with torch.no_grad():
 phase=out.conv(x).clamp(0,1)*255;expected=F.interpolate(F.pixel_shuffle(phase,4),scale_factor=3,mode='nearest');actual=out(x);error=float((actual-expected).abs().max());assert error<1e-12
print(json.dumps({'controls':rows,'output_decomposition_double_max_abs':error,'trained_quality_not_implied':True,'SDK_verified':False},indent=2))
