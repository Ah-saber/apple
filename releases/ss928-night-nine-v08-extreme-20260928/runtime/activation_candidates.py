"""Diagnostic affine activation8 pressure. No SDK bias/calibration equivalence."""
import json
from pathlib import Path
import torch
from torch import nn

def attach_pressure(model,scene,case,root,mode):
    data=json.loads((Path(root)/f'{scene}_activation_calibration_ranges.json').read_text())['results'][case]['ranges']
    if mode=='actweight8':
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    w=layer.weight.float();axes=(0,2,3) if isinstance(layer,nn.ConvTranspose2d) else (1,2,3);scale=w.abs().amax(axes,keepdim=True).clamp_min(1e-9)/127;layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
    def hook(item):
        lo,hi=item['min'],item['max'];scale=max((hi-lo)/255,1e-9);zero=max(0,min(255,round(-lo/scale)))
        def quantize(module,args):
            x=args[0];q=((x.float()/scale+zero).round().clamp(0,255)-zero)*scale;return (q.to(dtype=x.dtype),*args[1:])
        return quantize
    for name,layer in model.named_modules():
        if name in data and (mode!='actfront' or name.startswith('front.')):layer.register_forward_pre_hook(hook(data[name]))
    model.diagnostic_activation_pressure=True
    return model.eval()
