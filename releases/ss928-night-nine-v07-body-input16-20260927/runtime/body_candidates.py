"""Distilled body candidates retain the nine-frame front/reference/full output."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from continuation_candidates import load_continuation,prepare_inputs

BASE_CASE='combo_input9_half_aligned16_nearest'
OLD_RUN=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927')

class ShortBody(nn.Sequential):
    def __init__(self,depth,device='cpu',dtype=torch.float32):
        if depth not in (1,2,3):raise ValueError('Supported body depths: 1,2,3')
        layers=[]
        for _ in range(depth):
            layers.extend([nn.Conv2d(16,16,3,padding=1,device=device,dtype=dtype),nn.ReLU()])
        super().__init__(*layers)

@torch.no_grad()
def balance_body(body,passes=16):
    """Positive power-of-two scaling across adjacent Conv/ReLU channels.
    Only internal channels change; body output/reference/tail remain unscaled.
    Float arithmetic/SDK activation precision still needs separate validation.
    """
    convs=[layer for layer in body if isinstance(layer,nn.Conv2d)]
    for _ in range(passes):
        for first,second in zip(convs,convs[1:]):
            a=first.weight.double().abs().amax((1,2,3));b=second.weight.double().abs().amax((0,2,3));valid=(a>0)&(b>0)
            exponent=torch.where(valid,(.5*(b.clamp_min(1e-30).log2()-a.clamp_min(1e-30).log2())).round(),torch.zeros_like(a)).clamp(-4,4)
            scale=torch.pow(2.,exponent).to(first.weight.dtype)
            first.weight.mul_(scale[:,None,None,None]);first.bias.mul_(scale);second.weight.div_(scale[None,:,None,None])
    if not all(torch.isfinite(p).all() for p in body.parameters()):raise ValueError('Nonfinite equalized body')
    return body

class FusedHeadFirst(nn.Module):
    """Head4->16 k3 plus first body16->16 k3, with exact finite borders.
    Coefficients are formed in float64, then cast to inference precision.
    Float64 equivalence does not imply bit-identical float16 execution.
    """
    def __init__(self,head,first):
        import copy
        super().__init__();self.head=copy.deepcopy(head);self.first=copy.deepcopy(first)
        if head.weight.shape!=(16,4,3,3) or first.weight.shape!=(16,16,3,3):raise ValueError('Unexpected head/body geometry')
        for layer in (head,first):
            if layer.padding!=(1,1) or layer.stride!=(1,1) or layer.groups!=1:raise ValueError('Unexpected convolution attributes')
        h=head.weight.detach().cpu().double();b=first.weight.detach().cpu().double();w=torch.zeros((16,4,5,5),dtype=torch.float64)
        for by in range(3):
            for bx in range(3):
                for hy in range(3):
                    for hx in range(3):w[:,:,by+hy,bx+hx]+=b[:,:,by,bx]@h[:,:,hy,hx]
        bias=first.bias.detach().cpu().double()+b.sum((-2,-1))@head.bias.detach().cpu().double()
        self.fused=nn.Conv2d(4,16,5,padding=2,device=head.weight.device,dtype=head.weight.dtype)
        with torch.no_grad():self.fused.weight.copy_(w);self.fused.bias.copy_(bias)
    def forward(self,x):
        v=self.fused(x)
        top=self.first(self.head(x[:,:,:3,:])[:,:,:2,:])[:,:,:1,:]
        bottom=self.first(self.head(x[:,:,-3:,:])[:,:,-2:,:])[:,:,-1:,:]
        left=self.first(self.head(x[:,:,:,:3])[:,:,:,:2])[:,:,:,:1]
        right=self.first(self.head(x[:,:,:,-3:])[:,:,:,-2:])[:,:,:,-1:]
        middle=torch.cat((left[:,:,1:-1,:],v[:,:,1:-1,1:-1],right[:,:,1:-1,:]),-1)
        return F.relu(torch.cat((top,middle,bottom),-2))

def load_body(scene,case,run_dir,device='cuda'):
    root=Path(run_dir);old=root/'frozen_models'
    if not old.is_dir():old=OLD_RUN
    quant=None
    for suffix in ('_w8po','_w8pt'):
        if case.endswith(suffix):quant=suffix;case=case[:-len(suffix)]
    npu=case.endswith('_npu')
    if npu:case=case[:-4]
    balanced=case.endswith('_balanced')
    if balanced:case=case[:-9]
    if case=='head5_exact':
        model=load_continuation(scene,BASE_CASE,old,device=device);m=model.core.model
        fused=FusedHeadFirst(m.head[0],m.body[0].conv1.rep_conv)
        m.head=nn.Identity();m.body=nn.Sequential(fused,*list(m.body.children())[1:])
        return model.eval()
    def quantize_body(body):
        # Weight pressure control only; SDK activation/calibration is not modeled.
        with torch.no_grad():
            for layer in body.modules():
                if isinstance(layer,nn.Conv2d):
                    w=layer.weight.float();scale=(w.abs().amax((1,2,3),keepdim=True) if quant=='_w8po' else w.abs().max()).clamp_min(1e-9)/127
                    layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
    if not case.startswith('body'):
        model=load_continuation(scene,case,old,device=device)
        if quant:quantize_body(model.core.model.body)
        return model
    depth=int(case[4]);suffix=case[5:]
    if suffix not in ('_trained','_init','_pretrained','_preinit','_augtrained','_auginit','_quantsearch'):raise ValueError('Unknown body candidate '+case)
    model=load_continuation(scene,'combo_stats3_relu32_aligned16_half_output_fixed' if npu else BASE_CASE,old,device=device)
    name=f'{scene}_body{depth}_quantsearch.pt' if suffix=='_quantsearch' else f'{scene}_body{depth}_aug032000.pt' if '_aug' in suffix else f'{scene}_body{depth}_pre008000.pt' if '_pre' in suffix else f'{scene}_body{depth}_004000.pt'
    checkpoint=torch.load(root/name,map_location='cpu',weights_only=True)
    body=ShortBody(depth,device=device,dtype=model.core.model.head[0].weight.dtype)
    body.load_state_dict(checkpoint['body' if suffix in ('_trained','_pretrained','_augtrained','_quantsearch') else 'initial_body'],strict=True)
    if balanced:body=balance_body(body.float()).to(dtype=model.core.model.head[0].weight.dtype)
    if quant:quantize_body(body)
    model.core.model.body=body.to(memory_format=torch.channels_last)
    return model.eval()
