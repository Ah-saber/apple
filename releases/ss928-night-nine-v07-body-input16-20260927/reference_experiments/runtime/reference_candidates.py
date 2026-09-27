"""Compact 16-channel reference with teacher output supervision."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from body_candidates import load_body,BASE_CASE,prepare_inputs
BODY_RUN=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BODY-DISTILL-20260927')

class Reference16(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Conv2d(1,16,3,padding=1);self.middle=nn.Conv2d(16,16,3,padding=1);self.last=nn.Conv2d(16,16,3,padding=1);self.project=nn.Identity()
    def encode(self,x):
        v=F.relu(self.first(x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)));v=v+v.mean((-2,-1),keepdim=True)
        return self.last(F.relu(self.middle(v)))
    def forward(self,x):
        v=self.encode(x);return v+v.mean((-2,-1),keepdim=True)

class AlignedReference16(nn.Module):
    """Zero embedding of original 12-channel reference into 16 channels."""
    def __init__(self,original):
        import copy
        super().__init__()
        def pad_conv(old):
            if old.groups!=1 or old.in_channels not in (1,12) or old.out_channels not in (12,16):raise ValueError('Unsupported frozen reference convolution')
            new=nn.Conv2d(16 if old.in_channels==12 else old.in_channels,16,old.kernel_size,stride=old.stride,padding=old.padding,dilation=old.dilation,device=old.weight.device,dtype=old.weight.dtype)
            with torch.no_grad():new.weight.zero_();new.bias.zero_();new.weight[:old.out_channels,:old.in_channels].copy_(old.weight);new.bias[:old.out_channels].copy_(old.bias)
            return new
        def pad_sequence(sequence):return nn.Sequential(*(pad_conv(layer) if isinstance(layer,nn.Conv2d) else copy.deepcopy(layer) for layer in sequence))
        self.encoder=pad_sequence(original.encoder);self.pyramid=nn.ModuleList([pad_sequence(branch) for branch in original.pyramid]) if hasattr(original,'pyramid') else nn.ModuleList();self.project=pad_conv(original.project)
    def encode(self,x):
        v=self.encoder(x)
        for branch in self.pyramid:v=v+F.interpolate(branch(x),size=(64,64),mode='bilinear',align_corners=False)
        return v

def load_reference(scene,case,run_dir,device='cuda'):
    root=Path(run_dir);body_root=root/'body_models'
    if (root/f'{scene}_body3_aug032000.pt').is_file():body_root=root
    elif not body_root.is_dir():body_root=BODY_RUN
    if not case.startswith('ref16'):return load_body(scene,case,body_root,device=device)
    npu=case.endswith('_npu');stem=case[:-4] if npu else case
    if stem not in ('ref16_trained','ref16_init','ref16_body3_trained','ref16_body2_trained','ref16_aligned','ref16_aligned_body3','ref16_aligned_body2','ref16_aligned_body3_balanced','ref16_aligned_body2_balanced'):raise ValueError('Unknown reference16 case')
    body_case=('body3_augtrained' if 'body3' in case else 'body2_augtrained' if 'body2' in case else 'combo_stats3_relu32_aligned16_half_output_fixed' if npu else BASE_CASE)
    if 'balanced' in stem:body_case+='_balanced'
    if npu and body_case.startswith('body'):body_case+='_npu'
    model=load_body(scene,body_case,body_root,device=device);reference=Reference16()
    if 'aligned' in stem:
        model.core.model.global_reference=AlignedReference16(model.core.model.global_reference).to(memory_format=torch.channels_last)
        return model.eval()
    state=torch.load(root/f'{scene}_reference16_016000.pt',map_location='cpu',weights_only=True)
    reference.load_state_dict(state['initial_reference' if stem=='ref16_init' else 'reference'],strict=True)
    model.core.model.global_reference=reference.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
    return model.eval()
