"""Check training crop sampling against actual full-size interpolation."""
import json
import torch
from torch.nn import functional as F
torch.set_num_threads(2);torch.manual_seed(2785);rows=[]
for dtype in (torch.float64,torch.float32):
    ref=torch.randn(1,2,64,64,dtype=dtype);full=F.interpolate(ref,size=(512,640),mode='bilinear',align_corners=False)
    for y,x in ((0,0),(128,384),(448,576)):
        gy,gx=torch.meshgrid(torch.arange(y,y+64,dtype=dtype),torch.arange(x,x+64,dtype=dtype),indexing='ij');grid=torch.stack((2*(gx+.5)/640-1,2*(gy+.5)/512-1),-1)[None]
        sampled=F.grid_sample(ref,grid,padding_mode='border',align_corners=False);error=float((sampled-full[:,:,y:y+64,x:x+64]).abs().max());assert error<(1e-12 if dtype==torch.float64 else 2e-5)
        rows.append({'dtype':str(dtype),'feature_crop_yx':[y,x],'max_abs':error})
print(json.dumps({'controls':rows,'deployment_grid_sample':False,'includes_borders':True,'float16_bit_identity_claimed':False},indent=2))
