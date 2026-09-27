import sys,json
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import load_deployment
from extreme_candidates import PreservedFront
r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');torch.set_num_threads(2);torch.manual_seed(2807);rows=[]
for scene in ('ordinary','special'):
 old=load_deployment(scene,'body3_quantsearch',r,device='cpu').front.double()
 for h,w in [(8,12),(32,48)]:
  x=torch.randn(1,9,h,w,dtype=torch.float64)
  stats=F.pixel_unshuffle(F.conv2d(x,old.temporal),2);current,prev,recent=stats[:,:4],stats[:,4:8],stats[:,8:];z=torch.cat(((current-prev)*64,(current-recent)*64,current),1);reference=current+.025*old.last(F.relu(old.first(z)))
  for kind in ('raw','mean','sum'):
   for kernel in (6,7):
    with torch.no_grad():actual=PreservedFront(old,kind,kernel)(x);error=float((actual-reference).abs().max())
    assert error<1e-9;rows.append({'scene':scene,'kind':kind,'kernel':kernel,'shape':[h,w],'all_pixels_float64_max_abs':error})
(r/'preserved_front_proofs.json').write_text(json.dumps({'seed':2807,'checks':rows,'mathematical_equality_only_not_Half_bit_equality':True,'SDK_verified':False},indent=2));print('PRESERVED_PROOFS',len(rows))
