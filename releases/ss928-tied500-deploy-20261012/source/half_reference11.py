"""Keep separable reference interpolation in FP16 instead of FP32."""
import copy,torch
from torch import nn

class HalfBlockSample(nn.Module):
    def __init__(self,source):
        super().__init__();self.y=copy.deepcopy(source.y).half();self.x=copy.deepcopy(source.x).half()
    def forward(self,value):
        value=value.to(torch.float16)
        y=self.y(value.permute(0,2,1,3)).permute(0,2,1,3)
        return self.x(y.permute(0,3,2,1)).permute(0,3,2,1)

def convert(model):
    candidate=copy.deepcopy(model);changed=[]
    def visit(parent,prefix=''):
        for name,child in list(parent.named_children()):
            path=prefix+'.'+name if prefix else name
            if child.__class__.__name__=='BlockSample':
                setattr(parent,name,HalfBlockSample(child));changed.append(path)
            else:visit(child,path)
    visit(candidate)
    if not changed:raise ValueError('Existing block reference interpolation required')
    return candidate.eval().requires_grad_(False),changed
