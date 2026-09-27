import sys,json
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import load_deployment,SplitReferenceQuarter
r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');torch.set_num_threads(2);torch.manual_seed(2809);rows=[]
for scene in ['ordinary','special']:
 old=load_deployment(scene,'quarter_w32_3x3_body1_stage1_fusedtail',r,device='cpu').double();new=SplitReferenceQuarter(old)
 for h,w in [(16,24),(64,96)]:
  x=torch.randn(1,9,h,w,dtype=torch.float64)*.1+.3;ctx=torch.randn(1,1,64,64,dtype=torch.float64)*.1+.3
  with torch.no_grad():a=old(x,ctx);b=new(x,ctx);error=float((a-b).abs().max())
  assert error<1e-9;rows.append({'scene':scene,'shape':[h,w],'all_pixels_float64_gray_max_abs':error,'constant_channel_preserves_borders':True})
(r/'reference_split_proofs.json').write_text(json.dumps({'seed':2809,'checks':rows,'resize_logical_channels':{'old':32,'new':16},'mathematical_equality_only_not_Half_bit_identity':True,'SDK_verified':False},indent=2));print('REFSPLIT_PROOFS',len(rows))
