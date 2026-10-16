"""Run temporal-only body81 after the original-input parent77 stream."""
import json
from collections import deque
import numpy as np
import torch
from target_residual import ROOT
from quality_c3278 import QualityC3278
from body_temporal81 import BodyTemporal81,history_fields76
from body_temporal83 import BodyTemporal83

class QualityC3283:
    def __init__(self,number=81):
        if number not in [81,82,83]:raise ValueError('Supported rounds: 81, 82, 83')
        self.number=number;self.scene='weather_heavy_c32'
        self.parent=QualityC3278(77,export_features=True)
        model_type=BodyTemporal83 if number==83 else BodyTemporal81
        weights_number=83 if number==83 else 81
        self.model=model_type().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(ROOT/('MOTION%d-20261014'%weights_number)/self.scene/'last_step500.pt',map_location='cpu',weights_only=False)['model'])
        fit=json.loads((ROOT/('MOTION%d-20261014'%number)/self.scene/'fit.json').read_text())
        self.alpha=(fit['best'] if fit['status']=='passed' else fit['diagnostic_best'])['alpha']
        self.source='Original SR3 -> quality77 -> temporal-only body%d weights%d'%(number,weights_number)
        self.reset()
    def reset(self):self.bank=deque(maxlen=64);self.previous=None
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        if initial:self.reset()
        source=self.parent.native(fields,row,frame,initial=initial)
        observation=fields['current_only'][0,0,176:654].float().cpu().numpy()
        values=np.stack(list(self.bank)+[observation]);i=len(values)-1
        image=source[0,0].cpu().numpy();reference=image[176:654].copy();reference[:,:3]=reference[:,3:6][:,::-1]
        v=dict(self.parent.last_body_fields)
        v['history76']=history_fields76(values,i,(0,0,478,1280),reference,repair_boundary=True)
        s=torch.from_numpy(reference)[None,None].cuda()
        correction=self.model(v,s,s if self.previous is None else self.previous)[0,0].cpu().numpy()
        result=image.copy();result[176:654]+=self.alpha*correction
        self.bank.append(observation.copy());self.previous=s
        return torch.from_numpy(result.clip(0,255))[None,None].cuda()
