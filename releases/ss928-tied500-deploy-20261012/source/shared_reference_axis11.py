"""Calculate identical frozen reference windows once and reuse their outputs."""
import torch
from torch import nn
from torch.nn import functional as F

class SharedBlockAxis(nn.Module):
    def __init__(self,source):
        super().__init__();self.ranges=[];weights=[];self.mapping=[]
        for window,weight in zip(source.ranges,source.weights):
            found=next((i for i,(known,value) in enumerate(zip(self.ranges,weights)) if tuple(window)==tuple(known) and torch.equal(weight,value)),None)
            if found is None:
                found=len(weights);self.ranges.append(tuple(window));weights.append(weight.detach().clone())
            self.mapping.append(found)
        self.weights=nn.ParameterList([nn.Parameter(w,requires_grad=False) for w in weights])
        self.original_convolutions=len(source.weights)
    def forward(self,value):
        unique=[F.conv2d(value[:,lo:hi],w) for (lo,hi),w in zip(self.ranges,self.weights)]
        parts=[unique[i] for i in self.mapping]
        return parts[0] if len(parts)==1 else torch.cat(parts,1)

def convert(model):
    changes=[]
    def visit(parent,prefix=''):
        for name,child in list(parent.named_children()):
            path=prefix+'.'+name if prefix else name
            if child.__class__.__name__=='BlockAxis':
                candidate=SharedBlockAxis(child)
                if len(candidate.weights)<len(child.weights):
                    setattr(parent,name,candidate)
                    changes.append(dict(module=path,original_convolutions=len(child.weights),shared_convolutions=len(candidate.weights),output_mapping=candidate.mapping,selection='Exact equal frozen coefficients and equal input window'))
            else:visit(child,path)
    visit(model)
    return model,changes
