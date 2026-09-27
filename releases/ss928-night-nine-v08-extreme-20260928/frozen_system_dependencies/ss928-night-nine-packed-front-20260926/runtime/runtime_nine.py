"""Load the frozen fused graph without the training repository or original checkpoint."""
import types
from pathlib import Path
import torch
from model_base import RT4KSRB0,architecture_options
from collapse_output_shuffles import collapse_output_shuffles
from trajectory_batched_stats import trajectory_features_batched_stats
from fused_nine import FusedNine

def load_model(path,device='cuda',execution='source'):
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    if state['format']!='ss928_nine_fused_v1':raise ValueError('Unknown model artifact')
    c=state['architecture']
    base=RT4KSRB0(c['channels'],c['blocks'],deploy=True,global_reference=c['global_reference'],**architecture_options(c))
    base=collapse_output_shuffles(base)
    base._trajectory_features=types.MethodType(trajectory_features_batched_stats,base)
    model=FusedNine(base,raw_basis=True,channels_last=True,dense_second=True)
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device).eval()
    if execution=='compiled':
        if str(device).startswith('cpu'):raise ValueError('Compiled mode requires the verified CUDA environment')
        model=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
    elif execution!='source':raise ValueError('Unknown execution mode')
    return model
