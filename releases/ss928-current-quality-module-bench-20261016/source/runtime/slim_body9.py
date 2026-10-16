"""Prune hidden channels by training covariance and compensate kept convolutions."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from compact_student import owner,front_feature,tile_forward,CurriculumConv


def canonicalize(model):
    q=owner(model)
    for name,m in list(q.body.named_children()):
        if not isinstance(m,CurriculumConv):continue
        a=m.compact
        if not hasattr(a,'vertical'):raise ValueError('Expected retained axial body')
        v,h=a.vertical,a.horizontal
        new=nn.Conv2d(v.in_channels,h.out_channels,3,padding=1,bias=h.bias is not None,
                     device=v.weight.device,dtype=v.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(torch.einsum('orx,riy->oiyx',h.weight.double()[:,:,0,:],v.weight.double()[:,:,:,0]))
            if h.bias is not None:new.bias.copy_(h.bias)
        q.body._modules[name]=new
    return model


def project_of(model):
    if model.kind=='special':return None
    return owner(model).global_reference.project if model.kind else model.board.reference.reference.project


def layers(model):
    q=owner(model);out=(model.board.source.output if model.kind else q.output).conv
    result=[]
    for group in ([q.head] if model.kind else [q.front])+[q.body,q.tail]:
        result.extend(m for m in group.modules() if isinstance(m,nn.Conv2d))
    return result+[out]


def choose(cov,weight,width):
    residual=cov.clone().double()
    matrix=weight.double().permute(0,2,3,1).reshape(-1,weight.shape[1])
    selected=[]
    for _ in range(width):
        diagonal=residual.diag().clamp_min(1e-12)
        scores=(matrix@residual).square().sum(0)/diagonal
        if selected:scores[selected]=-1
        index=int(scores.argmax());selected.append(index)
        column=residual[:,index:index+1]
        residual-=column@column.T/diagonal[index]
        residual=(residual+residual.T)/2
    return sorted(selected)


def build_plan(reference,values,width_fraction=.5):
    model=canonicalize(copy.deepcopy(reference).float()).requires_grad_(False)
    chain=layers(model);names={id(m):n for n,m in model.named_modules()}
    sums={};mass={};handles=[];mask=None
    def make_hook(name):
        def hook(module,args):
            x=args[0].detach().float()
            support=F.interpolate(mask,size=x.shape[-2:],mode='area')
            w=1+32*F.max_pool2d((support>0).float(),5,1,2)
            x=x[...,::2,::2].flatten(2).permute(1,0,2).reshape(x.shape[1],-1).double()
            w=w[...,::2,::2].flatten().double()
            part=(x*w)@x.T
            sums[name]=part if name not in sums else sums[name]+part
            mass[name]=mass.get(name,0.)+float(w.sum())
        return hook
    for m in chain:handles.append(m.register_forward_pre_hook(make_hook(names[id(m)])))
    with torch.inference_mode():
        for value in values:
            mask=((value['target'].abs()+value['ghost'].abs())>0).float().cuda()
            raw=value['raw'].cuda();feature=front_feature(model,raw)
            tile_forward(model,feature,value['field'].cuda(),value['prior'].cuda())
    for h in handles:h.remove()
    covs={n:(x/mass[n]).cpu() for n,x in sums.items()}
    body_last=next(reversed([m for m in owner(model).body.modules() if isinstance(m,nn.Conv2d)]))
    plan=dict(layers=[],body_last_path=names[id(body_last)],source='Training RAW covariance with target/departure support weighted 33x',
              source_split='train',canonical_axial='Real-arithmetic merge; FP16 rounding changes',input_frames_unchanged=True,width_fraction=width_fraction)
    input_indices=list(range(chain[0].in_channels))
    for j,m in enumerate(chain):
        name=names[id(m)]
        if j==len(chain)-1:outputs=list(range(m.out_channels))
        else:
            nxt=chain[j+1];width=max(8,int(m.out_channels*width_fraction))
            outputs=choose(covs[names[id(nxt)]],nxt.weight.detach().cpu(),width)
        cov=covs[name]
        if len(input_indices)==m.in_channels:regression=torch.eye(m.in_channels,dtype=torch.float64)
        else:
            selected=cov[input_indices][:,input_indices]
            ridge=max(1e-10,float(selected.diag().mean())*1e-6)
            regression=torch.linalg.solve(selected+torch.eye(len(input_indices))*ridge,cov[:,input_indices].T).T
        plan['layers'].append(dict(path=name,input_indices=input_indices,output_indices=outputs,
                                   old_shape=list(m.weight.shape),regression=regression))
        if name==plan['body_last_path']:plan['body_indices']=outputs
        input_indices=outputs
    return plan


def rewrite(model,plan):
    canonicalize(model)
    for item in plan['layers']:
        path=item['path'];source=model.get_submodule(path);indices=item['output_indices'];regression=item['regression'].to(source.weight.device)
        new=nn.Conv2d(len(item['input_indices']),len(indices),source.kernel_size,stride=source.stride,
                     padding=source.padding,dilation=source.dilation,bias=source.bias is not None,
                     device=source.weight.device,dtype=source.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(torch.einsum('oihw,ir->orhw',source.weight[indices].double(),regression.double()))
            if source.bias is not None:new.bias.copy_(source.bias[indices])
        prefix,key=path.rsplit('.',1);model.get_submodule(prefix)._modules[key]=new
    project=project_of(model)
    if project is not None:
        indices=plan['body_indices'];new=nn.Conv2d(project.in_channels,len(indices),project.kernel_size,
            stride=project.stride,padding=project.padding,bias=project.bias is not None,
            device=project.weight.device,dtype=project.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(project.weight[indices])
            if project.bias is not None:new.bias.copy_(project.bias[indices])
        if model.kind:owner(model).global_reference.project=new
        else:model.board.reference.reference.project=new
    return model.requires_grad_(False)


def fitted_modules(model):
    q=owner(model)
    return ([q.head] if model.kind else [q.front])+[q.body,q.tail,(model.board.source.output if model.kind else q.output).conv]


def transfer(model,state):
    names={id(m):n for n,m in model.named_modules()}
    for module in fitted_modules(model):
        prefix=names[id(module)]+'.'
        module.load_state_dict({n[len(prefix):]:v for n,v in state.items() if n.startswith(prefix)},strict=True)


def batch_for(model,batch,plan):
    indices=plan['body_indices']
    batch['body_teacher']=batch['body_teacher'][:,indices]
    if model.kind!='special':
        batch['field']=batch['field'][:,indices]
        batch['pair_field']=batch['pair_field'][:,indices]
    return batch
