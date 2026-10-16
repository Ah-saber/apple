"""Restore the retained reference; no BORDERFIX main or correction weights."""
import copy,torch
from target_residual import ROOT
from train_nativewidth9 import previous
from phaseown_model11 import training_model as retained_model
from tieddecode_phase11 import TiedDecode,CurrentRouteBoard
from npu_graph_candidate import optimize

def training_model(scene,reference,count,kind):
    parent=retained_model(scene,reference,count,kind)
    branch=TiedDecode(parent.source_count).to(next(parent.parameters()).device,dtype=torch.float32)
    model=CurrentRouteBoard(parent.reference.requires_grad_(False),branch,parent.source_count)
    ck=torch.load(ROOT/('quietpair11_weights_'+scene)/'last_step500.pt',map_location='cpu',weights_only=False)
    if ck['steps']!=500 or ck['input_count']!=model.source_count:raise ValueError('Fixed QUIETPAIR500 parent required')
    names=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in ck['model'].items():
            if not name.startswith('branch.'):raise ValueError('Frozen main expected')
            names[name].copy_(value[:4] if name.startswith('branch.auxiliary.') else value)
    return model

def boards(scene,base,count,kind,box,step):
    original=previous(scene,base,count,kind,box)
    parent=training_model(scene,original,count,kind)
    candidate=copy.deepcopy(parent)
    ck=torch.load(ROOT/('tieddecode11_weights_'+scene)/('last_step'+str(step)+'.pt'),map_location='cpu',weights_only=False)
    if ck['steps']!=step or ck['input_count']!=candidate.source_count:raise ValueError('SpatialTime fixed screen checkpoint mismatch')
    names=dict(candidate.named_parameters())
    with torch.no_grad():
        for name,value in ck['model'].items():
            if not name.startswith('branch.'):raise ValueError('Reference learning is forbidden')
            names[name].copy_(value)
    results=[]
    from half_reference11 import convert
    for model in [parent,candidate]:
        model.half();model=optimize(model,kind)[0].eval().requires_grad_(False);model,_=convert(model);results.append(model)
    return tuple(results)

