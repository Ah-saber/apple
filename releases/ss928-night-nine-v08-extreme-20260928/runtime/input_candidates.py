"""16-slot input representations; statistics remain entirely in the model."""
import torch
from torch import nn
from body_candidates import load_body,prepare_inputs

class Input16System(nn.Module):
    def __init__(self,system,layout):
        super().__init__();self.system=system;self.input_channels=16;self.input_half=True;self.input_layout=layout
        w=system.front.weight
        if w.shape!=(32,9,3,3):raise ValueError('Require existing signed 32-channel statistics3 front')
        system.front.weight=torch.cat((w,torch.zeros_like(w[:,:7])),1)
    def forward(self,x,context):
        if self.input_layout=='native5':x=x.reshape(x.shape[0],x.shape[2],x.shape[3],16)
        x=x.permute(0,3,1,2)
        return self.system(x,context)

def load_input(scene,case,run_dir,device='cuda'):
    if not case.startswith('input16_'):return load_body(scene,case,run_dir,device=device)
    layout='native5' if 'native5' in case else 'nhwc16'
    if case not in ('input16_nhwc','input16_native5','input16_nhwc_body3','input16_native5_body3'):raise ValueError('Unknown input representation case')
    base='body3_quantsearch_npu' if 'body3' in case else 'combo_stats3_relu32_aligned16_half_output_fixed'
    return Input16System(load_body(scene,base,run_dir,device=device),layout).eval()
