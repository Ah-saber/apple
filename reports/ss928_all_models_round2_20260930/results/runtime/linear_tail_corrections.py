"""Linear 3x3+3x3 composition with explicit correction of intermediate zero padding."""
import torch
from torch import nn
from torch.nn import functional as F
from speed_variants import LinearTail


class CorrectedLinearTail(nn.Module):
    def __init__(self,tail,projection):
        super().__init__();fused=LinearTail(tail,projection);self.conv=fused.conv
        linear=tail[1].rep_conv if isinstance(tail,nn.Sequential) else tail
        t=linear.weight.detach().cpu().double();p=projection.weight.detach().cpu().double();b=linear.bias.detach().cpu().double()
        for side,py,ty in [('top',0,2),('bottom',2,0)]:
            w=torch.zeros(p.shape[0],t.shape[1],1,5,dtype=torch.float64)
            for px in range(3):
                for tx in range(3):w[:,:,0,px+tx]+=p[:,:,py,px]@t[:,:,ty,tx]
            layer=nn.Conv2d(t.shape[1],p.shape[0],(1,5),padding=(0,2),device=projection.weight.device,dtype=projection.weight.dtype)
            with torch.no_grad():layer.weight.copy_(w);layer.bias.copy_(p[:,:,py,:].sum(-1)@b)
            setattr(self,side,layer)
        for side,px,tx in [('left',0,2),('right',2,0)]:
            w=torch.zeros(p.shape[0],t.shape[1],5,1,dtype=torch.float64)
            for py in range(3):
                for ty in range(3):w[:,:,py+ty,0]+=p[:,:,py,px]@t[:,:,ty,tx]
            layer=nn.Conv2d(t.shape[1],p.shape[0],(5,1),padding=(2,0),device=projection.weight.device,dtype=projection.weight.dtype)
            with torch.no_grad():layer.weight.copy_(w);layer.bias.copy_(p[:,:,:,px].sum(-1)@b)
            setattr(self,side,layer)
        for name,py,px,ty,tx in [('tl',0,0,2,2),('tr',0,2,2,0),('bl',2,0,0,2),('br',2,2,0,0)]:
            self.register_buffer(name+'_weight',(p[:,:,py,px]@t[:,:,ty,tx]).reshape(p.shape[0],t.shape[1],1,1).to(projection.weight))
            self.register_buffer(name+'_bias',(p[:,:,py,px]@b).to(projection.weight))

    def corner(self,x,name):return F.conv2d(x,getattr(self,name+'_weight'),getattr(self,name+'_bias'))

    def forward(self,x):
        v=self.conv(x);top=v[:,:,:1,:]-self.top(x[:,:,:1,:]);bottom=v[:,:,-1:,:]-self.bottom(x[:,:,-1:,:])
        left=self.left(x[:,:,:,:1]);right=self.right(x[:,:,:,-1:])
        top=torch.cat((top[:,:,:,:1]-left[:,:,:1,:]+self.corner(x[:,:,:1,:1],'tl'),top[:,:,:,1:-1],
                       top[:,:,:,-1:]-right[:,:,:1,:]+self.corner(x[:,:,:1,-1:],'tr')),-1)
        bottom=torch.cat((bottom[:,:,:,:1]-left[:,:,-1:,:]+self.corner(x[:,:,-1:,:1],'bl'),bottom[:,:,:,1:-1],
                          bottom[:,:,:,-1:]-right[:,:,-1:,:]+self.corner(x[:,:,-1:,-1:],'br')),-1)
        middle=torch.cat((v[:,:,1:-1,:1]-left[:,:,1:-1,:],v[:,:,1:-1,1:-1],v[:,:,1:-1,-1:]-right[:,:,1:-1,:]),-1)
        return torch.cat((top,middle,bottom),-2)
