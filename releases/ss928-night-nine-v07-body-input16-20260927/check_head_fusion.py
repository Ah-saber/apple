"""Bounded float64 finite-image check including every border pixel."""
import json,sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from body_candidates import FusedHeadFirst
torch.set_num_threads(2);torch.manual_seed(2784);results=[]
for h,w in ((3,4),(8,11),(64,96)):
 head=nn.Conv2d(4,16,3,padding=1,dtype=torch.float64);first=nn.Conv2d(16,16,3,padding=1,dtype=torch.float64);x=torch.randn(1,4,h,w,dtype=torch.float64)
 with torch.no_grad():
  fused=FusedHeadFirst(head,first);expected=torch.relu(first(head(x)));actual=fused(x);error=float((expected-actual).abs().max());assert error<1e-12
 results.append({'shape':[h,w],'all_pixels_max_abs':error})
print(json.dumps({'float64_equivalence_only':True,'includes_full_borders':True,'results':results},indent=2))
