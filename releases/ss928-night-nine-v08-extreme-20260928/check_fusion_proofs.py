import json,sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from deployment_candidates import MappedOutput
torch.set_num_threads(2);torch.manual_seed(2802);results=[]
for h,w in [(2,3),(9,11),(16,24)]:
 tail=nn.Conv2d(32,16,3,padding=1).double();projection=nn.Conv2d(16,16,1).double();x=torch.randn(1,32,h,w,dtype=torch.float64);fused=nn.Conv2d(32,16,3,padding=1).double()
 with torch.no_grad():
  m=projection.weight[:,:,0,0];fused.weight.copy_(torch.einsum('oi,ickl->ockl',m,tail.weight));fused.bias.copy_(m@tail.bias+projection.bias);error=float((fused(x)-projection(tail(x))).abs().max())
 assert error<1e-10;results.append({'proof':'quarter_tail3_projection1_all_pixels','shape':[h,w],'max_abs':error})
for factor in (2,4):
 for h,w in [(2,3),(9,11),(16,24)]:
  old=nn.Module();old.conv=nn.Conv2d(16,16,3,padding=1).double();x=torch.randn(1,16,h,w,dtype=torch.float64)
  with torch.no_grad():
   reference=MappedOutput(old,factor,'pixel')(x)
   for mode in ('deconv','cascade'):
    error=float((MappedOutput(old,factor,mode)(x)-reference).abs().max());assert error<1e-10;results.append({'proof':'output_mapping_'+mode,'factor':factor,'shape':[h,w],'max_abs':error})
Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927/fusion_mapping_proofs.json').write_text(json.dumps({'seed':2802,'dtype':'float64','checks':results,'SDK_verified':False,'float16_bit_equivalence_not_claimed':True},indent=2));print('FUSION_MAPPING_PROOFS',len(results))
