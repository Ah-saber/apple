"""Random float64 proof of internal positive channel rescaling."""
import sys,json,copy
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from body_candidates import ShortBody,balance_body
torch.set_num_threads(2);torch.manual_seed(2784);rows=[]
for depth in (2,3):
    model=ShortBody(depth,dtype=torch.float64)
    with torch.no_grad():model[0].weight[0]*=16;model[2].weight[:,0]/=16
    balanced=balance_body(copy.deepcopy(model))
    for h,w in ((3,4),(12,14),(64,96)):
        x=torch.randn(1,16,h,w,dtype=torch.float64)
        with torch.no_grad():error=float((model(x)-balanced(x)).abs().max())
        assert error<1e-12;rows.append({'depth':depth,'shape':[h,w],'all_pixel_max_abs':error})
print(json.dumps({'float64_equivalence_only':True,'controls':rows,'SDK_verified':False},indent=2))
