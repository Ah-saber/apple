"""Load saved approximation weights; requires the frozen deployment runtime on sys.path."""
import hashlib
from pathlib import Path
import torch
from torch import nn

class FactorizedLayer(nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.horizontal=nn.Conv2d(13,rank,(1,5),padding=(0,2),bias=False,dtype=torch.float16)
        self.vertical=nn.Conv2d(rank,16,(5,1),padding=(2,0),dtype=torch.float16)
        self.to(memory_format=torch.channels_last)
    def forward(self,x):return self.vertical(self.horizontal(x))

def load_candidate(candidate,baseline,device='cuda',execution='source',remove_zero_init=False):
    from runtime_nine import load_model
    baseline=Path(baseline)
    state=torch.load(candidate,map_location='cpu',weights_only=True)
    if state['format']!='ss928_nine_factorized_candidate_v1':raise ValueError('Unexpected artifact format')
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=state['baseline_artifact_sha256']:raise ValueError('Wrong frozen baseline')
    model=load_model(baseline,device='cpu',execution='source')
    model.first=FactorizedLayer(state['rank'])
    model.load_state_dict(state['model'],strict=True)
    if remove_zero_init:
        import types
        from trajectory_nozero import trajectory_features_nozero
        model.model._trajectory_features=types.MethodType(trajectory_features_nozero,model.model)
    model=model.to(device).eval()
    if execution=='compiled':model=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
    elif execution!='source':raise ValueError('Unexpected execution mode')
    return model
