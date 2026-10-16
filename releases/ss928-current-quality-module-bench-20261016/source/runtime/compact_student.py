"""Full-width axial or grouped spatial layers; preserve outer activation."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class Axial(nn.Module):
    def __init__(self,source):
        super().__init__();o,i,h,w=source.weight.shape;rank=o
        matrix=source.weight.detach().double().permute(0,3,1,2).reshape(o*w,i*h)
        u,s,v=torch.linalg.svd(matrix,full_matrices=False)
        a=(s[:rank].sqrt()[:,None]*v[:rank]).reshape(rank,i,h,1)
        b=(u[:,:rank]*s[:rank].sqrt()[None]).reshape(o,w,rank).permute(0,2,1)[:,:,None,:]
        kw=dict(device=source.weight.device,dtype=source.weight.dtype)
        self.vertical=nn.Conv2d(i,rank,(3,1),padding=(1,0),bias=False,**kw)
        self.horizontal=nn.Conv2d(rank,o,(1,3),padding=(0,1),bias=source.bias is not None,**kw)
        with torch.no_grad():
            self.vertical.weight.copy_(a);self.horizontal.weight.copy_(b)
            if source.bias is not None:self.horizontal.bias.copy_(source.bias)
        self.energy=float(s[:rank].square().sum()/s.square().sum())
    def forward(self,value):return self.horizontal(self.vertical(value))

class GroupedSpatial(nn.Module):
    def __init__(self,source,groups=2):
        super().__init__();o,i,h,w=source.weight.shape;assert o%groups==i%groups==0
        kw=dict(device=source.weight.device,dtype=source.weight.dtype)
        self.spatial=nn.Conv2d(i,o,3,padding=1,groups=groups,bias=False,**kw)
        self.mix=nn.Conv2d(o,o,1,bias=source.bias is not None,**kw)
        energies=[]
        with torch.no_grad():
            for group in range(groups):
                start,end=group*(i//groups),(group+1)*(i//groups)
                matrix=source.weight[:,start:end].detach().double().flatten(1)
                u,s,v=torch.linalg.svd(matrix,full_matrices=False);rank=o//groups
                self.spatial.weight[group*rank:(group+1)*rank].copy_(v[:rank].reshape(rank,i//groups,3,3))
                self.mix.weight[:,group*rank:(group+1)*rank,0,0].copy_(u[:,:rank]*s[:rank])
                energies.append(float(s[:rank].square().sum()/s.square().sum()))
            if source.bias is not None:self.mix.bias.copy_(source.bias)
        self.energy=energies
    def forward(self,value):return self.mix(self.spatial(value))

class CurriculumConv(nn.Module):
    def __init__(self,source,mode):
        super().__init__();self.original=copy.deepcopy(source).requires_grad_(False)
        self.compact=Axial(source) if mode=='axial' else GroupedSpatial(source)
        self.alpha=1.
    def forward(self,value):
        new=self.compact(value)
        if self.alpha>=1:return new
        return self.original(value)*(1-self.alpha)+new*self.alpha

def owner(model):return model.board.source.core.model if model.kind else model.board.base.quarter

def rewrite(model,mode):
    records=[];body=owner(model).body
    for name,child in list(body.named_children()):
        if isinstance(child,nn.Conv2d) and child.kernel_size==(3,3) and child.stride==(1,1) and child.groups==1:
            value=CurriculumConv(child,mode);body._modules[name]=value
            old=child.weight.numel();new=sum(p.numel() for n,p in value.compact.named_parameters() if 'weight' in n)
            records.append(dict(module=name,mode=mode,channels=[child.in_channels,child.out_channels],old_coefficients=old,
                new_coefficients=new,initialization_energy=value.compact.energy,activation_preserved=True))
    if not records:raise ValueError('No supported body layers')
    return records

def set_alpha(model,alpha):
    for m in owner(model).body.modules():
        if isinstance(m,CurriculumConv):m.alpha=float(alpha)

def tile_forward(model,front,field,prior):
    value=owner(model);body=value.body(front)
    if model.kind:
        source=model.board.source
        phases=source.output.conv(value.tail(body+field)) if model.kind=='ordinary' else source.output.conv(value.tail(body))+field
    else:phases=value.output.conv(value.tail(body+field))
    return F.pixel_shuffle(phases+prior,2),body

def front_feature(reference,raw):
    m=reference.board;raw=raw[:,:reference.source_count]
    if reference.kind:
        s=m.source;value=s.core.half_input(s.front(raw.to(s.front.temporal_conv.weight.dtype)))
        return s.core.model.head(value.to(next(s.core.model.head.parameters()).dtype))
    q=m.base.quarter;return q.front(raw.to(q.front.first.weight.dtype))

def transfer_learned(model,state,mode):
    """Keep destination ROI sampling buffers; transfer only fitted core components."""
    if mode=='temporal':
        conv=model.board.source.front.temporal_conv if model.kind else owner(model).front.first
        name=next(n for n,p in model.named_parameters() if p is conv.weight)
        with torch.no_grad():conv.weight.copy_(state[name])
        return [name]
    value=owner(model)
    modules=[value.body,value.tail,(model.board.source.output if model.kind else value.output).conv]
    names={id(m):n for n,m in model.named_modules()};copied=[]
    for module in modules:
        prefix=names[id(module)]+'.'
        selected={n[len(prefix):]:v for n,v in state.items() if n.startswith(prefix)}
        module.load_state_dict(selected,strict=True);copied.extend(prefix+n for n in selected)
    return copied
