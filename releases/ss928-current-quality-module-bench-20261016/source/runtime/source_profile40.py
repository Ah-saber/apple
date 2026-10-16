"""Load the preserved per-scene source for the regional tone experiment."""
import json
import torch
from target_residual import ROOT
from repair_motion15 import Repair

def source_model(scene):
    parameters=None
    if scene=='night_special':
        from raw_stable27 import RawStable27
        model=RawStable27();parent='MOTION27-20261013'
    elif scene=='weather_heavy':
        from contrast_guard18 import OUT18
        parameters=json.loads((OUT18/(scene+'-fit.json')).read_text())['best']['parameters']
        if parameters['parent']=='repair17':
            from repair_gate17 import Repair17
            model=Repair17();parent='MOTION17-20261013'
        else:model=Repair();parent='MOTION15-20261013'
    else:model=Repair();parent='MOTION15-20261013'
    ck=torch.load(ROOT/parent/scene/'last_step500.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'])
    return model,parameters,parent
