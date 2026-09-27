"""Actual frozen-reference zero-embedding equivalence, signed inputs included."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from reference_candidates import AlignedReference16,BODY_RUN
from body_candidates import load_body,BASE_CASE
torch.set_num_threads(2);torch.manual_seed(2785);rows=[]
for scene in ('ordinary','special'):
    original=load_body(scene,BASE_CASE,BODY_RUN,device='cpu').core.model.global_reference.double();aligned=AlignedReference16(original)
    for kind in ('signed_random','constant','corner_impulse'):
        x=torch.randn(1,1,64,64,dtype=torch.float64) if kind=='signed_random' else torch.full((1,1,64,64),.3,dtype=torch.float64)
        if kind=='corner_impulse':x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-1
        with torch.no_grad():
            r=original.encode(x);expected=original.project(r+r.mean((-2,-1),keepdim=True));r=aligned.encode(x);actual=aligned.project(r+r.mean((-2,-1),keepdim=True));error=float((actual-expected).abs().max())
        assert error<1e-12;rows.append({'scene':scene,'input':kind,'all_pixel_max_abs':error})
print(json.dumps({'actual_frozen_coefficients':True,'float64_equivalence_only':True,'controls':rows,'SDK_verified':False},indent=2))
