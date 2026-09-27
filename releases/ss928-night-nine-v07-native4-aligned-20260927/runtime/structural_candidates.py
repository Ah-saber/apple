"""New complete-model candidates; no claims of SDK compatibility or NPU speed."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from joint_candidates import load_joint

class SpaceStatisticsFront(nn.Module):
    def __init__(self, original, mode):
        super().__init__();self.mode=mode
        w=original.stats_weight
        temporal=torch.stack([w[4*g,:,0,0] for g in range(3)])[:,:,None,None]
        self.register_buffer('temporal',temporal.clone())
        packed=torch.zeros((12,36,1,1),device=w.device,dtype=w.dtype)
        for g in range(3):
            for phase in range(4):packed[g*4+phase,phase::4,0,0]=temporal[g,:,0,0]
        self.register_buffer('packed',packed)
        self.first=copy.deepcopy(original.first);self.last=copy.deepcopy(original.last)
    def forward(self,x):
        x=x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        if self.mode=='space_pack':stats=F.conv2d(F.pixel_unshuffle(x,2),self.packed)
        else:stats=F.pixel_unshuffle(F.conv2d(x,self.temporal),2)
        current,old,recent=stats[:,:4].float(),stats[:,4:8].float(),stats[:,8:].float()
        v=torch.cat(((current-old)*64,(current-recent)*64,current),1).to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        return current+self.last(F.relu(self.first(v))).float()*.025

class AxisPermutation(nn.Module):
    def __init__(self,order='horizontal',device='cpu'):
        super().__init__();self.order=order
        first=torch.zeros((36,6,1,6) if order=='horizontal' else (36,6,6,1),device=device,dtype=torch.float16)
        second=torch.zeros((6,1,6,1) if order=='horizontal' else (6,1,1,6),device=device,dtype=torch.float16)
        for dy in range(6):
            for dx in range(6):
                if order=='horizontal':first[dy*6+dx,dy,0,dx]=1
                else:first[dy*6+dx,dx,dy,0]=1
        for p in range(6):
            if order=='horizontal':second[p,0,p,0]=1
            else:second[p,0,0,p]=1
        self.register_buffer('first',first);self.register_buffer('second',second)
    def forward(self,x):
        strides=((1,6),(6,1)) if self.order=='horizontal' else ((6,1),(1,6))
        y=F.conv_transpose2d(x,self.first.to(x.dtype),stride=strides[0])
        return F.conv_transpose2d(y,self.second.to(x.dtype),stride=strides[1])

class ReducedPhaseOutput(nn.Module):
    def __init__(self,original,factor,mode='nearest'):
        super().__init__();self.factor=factor;self.mode=mode;self.rest=6//factor
        dtype=torch.float32 if mode=='preserve_nearest_fp32' else original.weight.dtype
        self.conv=nn.Conv2d(16,factor*factor,3,padding=1,device=original.weight.device,dtype=dtype)
        w=original.weight.reshape(6,6,16,3,3);b=original.bias.reshape(6,6)
        with torch.no_grad():
            for y in range(factor):
                for x in range(factor):
                    self.conv.weight[y*factor+x].copy_(w[y*self.rest:(y+1)*self.rest,x*self.rest:(x+1)*self.rest].float().mean((0,1)))
                    self.conv.bias[y*factor+x].copy_(b[y*self.rest:(y+1)*self.rest,x*self.rest:(x+1)*self.rest].float().mean())
            if mode in ('preserve_nearest','preserve_nearest_fp32','preserve_fixed'):
                if factor!=3:raise ValueError('Native-mean constraint requires nine phases')
                from phase_projection import preserve_native_means
                source_weight=original.weight.detach().to(dtype);source_bias=original.bias.detach().to(dtype)
                ww,bb,self.constraint_proof=preserve_native_means(source_weight.cpu().numpy(),source_bias.cpu().numpy())
                self.conv.weight.copy_(torch.from_numpy(ww).to(self.conv.weight));self.conv.bias.copy_(torch.from_numpy(bb).to(self.conv.bias))
        if mode in ('fixed','preserve_fixed'):
            kernel=torch.zeros((factor*factor,1,6,6),device=original.weight.device,dtype=original.weight.dtype)
            for y in range(factor):
                for x in range(factor):kernel[y*factor+x,0,y*self.rest:(y+1)*self.rest,x*self.rest:(x+1)*self.rest]=1
            self.register_buffer('fixed_kernel',kernel)
    def forward(self,x):
        x=x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)
        if self.mode in ('fixed','preserve_fixed'):return F.conv_transpose2d(self.conv(x),self.fixed_kernel,stride=6)
        x=F.pixel_shuffle(self.conv(x),self.factor) if self.factor>1 else self.conv(x)
        mode='nearest' if self.mode.startswith('preserve_nearest') else self.mode
        return F.interpolate(x,scale_factor=self.rest,mode=mode,align_corners=False if mode=='bilinear' else None)

def replace_gelu(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.GELU):setattr(module,name,nn.ReLU())
        else:replace_gelu(child)

class StructuralSystem(nn.Module):
    def __init__(self,base,case):
        super().__init__();self.core=base.core;self.front=base.front
        self.permutation=None;self.output=None
        if 'space_pack' in case:self.front=SpaceStatisticsFront(base.front,'temporal_space_pack' if 'temporal_space_pack' in case else 'space_pack')
        if case.startswith('axis_'):self.permutation=AxisPermutation(case.split('_',1)[1],base.front.first.weight.device)
        for phase_case in ('phase3_quantsearch_fixed','phase3_quantsearch_nearest','phase3_preserve_fixed','phase3_k1_ridge_nearest','phase3_preserve_nearest_fp32','phase3_k1_nearest','phase3_preserve_nearest','phase3_nearest','phase3_bilinear','phase3_fixed','phase2_nearest','phase1_nearest'):
            if phase_case in case:
                factor=int(phase_case[5]);self.output=ReducedPhaseOutput(base.core.model.upsample[0],factor,('preserve_fixed' if phase_case.endswith('fixed') else 'preserve_nearest') if 'quantsearch' in phase_case else 'preserve_nearest' if 'k1' in phase_case else phase_case.split('_',1)[1]);break
        if 'reference_relu' in case:replace_gelu(self.core.model.global_reference)
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        features=m.body(m.head(c.half_input(self.front(stack))))
        ref=m.global_reference.encode(c.half_input(context))
        ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        features=m.tail(features+ref)
        if self.output is not None:gray=self.output(features)
        else:
            phases=m.upsample[0](features)
            gray=self.permutation(phases) if self.permutation is not None else F.pixel_shuffle(phases,6)
        return gray.float().clamp(0,1)*255

def load_structural(scene,case,run_dir,device='cuda'):
    rest=case
    if rest not in ('baseline','axis_horizontal','axis_vertical'):
        for prefix in ('temporal_space_pack','space_pack'):
            if rest==prefix or rest.startswith(prefix+'_'):
                rest=rest[len(prefix):].lstrip('_');break
        for prefix in ('reference_relu_output_stable','reference_relu_output','reference_relu'):
            if rest==prefix or rest.startswith(prefix+'_'):
                rest=rest[len(prefix):].lstrip('_');break
        allowed=('', 'phase3_quantsearch_fixed', 'phase3_quantsearch_nearest', 'phase3_preserve_fixed','phase3_k1_ridge_nearest','phase3_preserve_nearest_fp32','phase3_k1_nearest','phase3_preserve_nearest','phase3_nearest','phase3_bilinear','phase3_fixed','phase2_nearest','phase1_nearest')
        if rest not in allowed:raise ValueError('Unknown structural candidate: '+case)
    base=load_joint(scene,'baseline',run_dir,device=device)
    if case=='baseline':return base
    model=StructuralSystem(base,case).eval()
    if 'quantsearch' in case:
        checkpoint=torch.load(Path(run_dir)/f'{scene}_output_quantsearch.pt',map_location='cpu',weights_only=True)
        model.output.conv.load_state_dict(checkpoint['projection'],strict=True)
        model.output.constraint_proof=checkpoint['proof']
    if 'phase3_k1' in case:
        original=model.output.conv
        model.output.conv=nn.Conv2d(16,9,1,device=original.weight.device,dtype=original.weight.dtype)
        tag='output_k1_ridge' if 'k1_ridge' in case else 'output_k1'
        checkpoint=torch.load(Path(run_dir)/f'{scene}_{tag}_004000.pt',map_location='cpu',weights_only=True)
        model.output.conv.load_state_dict(checkpoint['projection'],strict=True)
        del model.output.constraint_proof
    if 'reference_relu' in case:
        ref_case='reference_relu_output_stable' if 'reference_relu_output_stable' in case else ('reference_relu_output' if 'reference_relu_output' in case else 'reference_relu')
        checkpoint=torch.load(Path(run_dir)/f'{scene}_{ref_case}_004000.pt',map_location='cpu',weights_only=True)
        model.core.model.global_reference.load_state_dict(checkpoint['reference'],strict=True)
        model.core.model.global_reference.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
    return model
