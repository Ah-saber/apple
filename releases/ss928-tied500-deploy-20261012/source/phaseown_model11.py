"""Load an original-background model with a fresh five-convolution display branch."""
import copy,torch
from target_residual import ROOT
from train_nativewidth9 import previous
from phaseown_phase11 import PhaseBasis,CurrentRouteBoard
from axial_front9 import RecentFront
from current_input11 import folded_conv
from npu_graph_candidate import optimize

def training_model(scene,reference,count,kind):
    reference=copy.deepcopy(reference).float().requires_grad_(False)
    input_count=min(5,count)
    if input_count!=count:
        reference.board.source.front=RecentFront(reference.board.source.front,input_count)
        reference.source_count=input_count
        if reference.accepted is not None:
            reference.accepted.conv=folded_conv(reference.accepted.conv,input_count)
            if hasattr(reference.accepted,'count'):reference.accepted.count=input_count
    branch=PhaseBasis(input_count).to(device=next(reference.parameters()).device,dtype=torch.float32)
    return CurrentRouteBoard(reference,branch,input_count)

def models(scene,base,count,kind,box):
    control=previous(scene,base,count,kind,box);candidate=training_model(scene,control,count,kind)
    ck=torch.load(ROOT/('phaseown11_weights_'+scene)/'last.pt',map_location='cpu',weights_only=False)
    if ck['steps']!=4000:raise ValueError('Fixed final4000 checkpoint required')
    parameters=dict(candidate.named_parameters())
    with torch.no_grad():
        for name,value in ck['model'].items():parameters[name].copy_(value)
    candidate.branch.fold_encoder();candidate.half()
    control,candidate=tuple(optimize(m,kind)[0].eval().requires_grad_(False) for m in [control,candidate])
    from half_reference11 import convert
    candidate,changed=convert(candidate)
    return control,candidate
