"""Complete-output variants; arithmetic changes are measured, never assumed exact."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from compact_model import FixedSample


class RepeatByteReorder(nn.Module):
    def __init__(self,factor,to_byte=True):
        super().__init__(); self.factor=factor;self.to_byte=to_byte

    def forward(self,phases):
        p=phases.to(torch.uint8) if self.to_byte else phases; n,c,h,w=p.shape; s=self.factor
        p=p.reshape(n,1,s,s,h,w).permute(0,1,4,2,5,3)
        return p.reshape(n,1,h,s,1,w,s,1).expand(n,1,h,s,3,w,s,3).reshape(n,1,h*s*3,w*s*3)


class NightByteOutput(nn.Module):
    def __init__(self,source):
        super().__init__();self.conv=copy.deepcopy(source.conv);self.output=RepeatByteReorder(2)

    def finish(self,phases):return self.output(phases)

    def forward(self,x):return self.finish(self.conv(x.to(self.conv.weight.dtype)).clamp(0,1)*255)


class NightByte(nn.Module):
    def __init__(self,source):
        super().__init__();self.source=copy.deepcopy(source);self.source.output=NightByteOutput(source.output)

    def forward(self,x,c):return self.source(x,c)


class HalfSample(FixedSample):
    def forward(self,x):
        x=x.half()
        y=x.index_select(2,self.y0)*(1-self.yw.half())+x.index_select(2,self.y1)*self.yw.half()
        return y.index_select(3,self.x0)*(1-self.xw.half())+y.index_select(3,self.x1)*self.xw.half()


class ConvSample(nn.Module):
    """Integer-scale align_corners=False bilinear interpolation with replicated borders."""
    def __init__(self,channels,height,width,source=64):
        super().__init__(); assert height%source==width%source==0
        self.channels=channels;self.sy=height//source;self.sx=width//source
        for name,s in [('ky',self.sy),('kx',self.sx)]:
            k=2*s-(s%2);v=(1-(torch.arange(k)-(k-1)/2).abs()/s)
            weight=v.reshape(1,1,k,1) if name=='ky' else v.reshape(1,1,1,k)
            self.register_buffer(name,weight.repeat(channels,1,1,1))
        self.register_buffer('border',torch.arange(-1,source+1).clamp(0,source-1))

    def forward(self,x):
        x=x.index_select(2,self.border)
        x=F.conv_transpose2d(x,self.ky.to(x),stride=(self.sy,1),padding=(self.sy//2,0),groups=self.channels)
        x=x[:,:,self.sy:-self.sy,:].index_select(3,self.border)
        x=F.conv_transpose2d(x,self.kx.to(x),stride=(1,self.sx),padding=(0,self.sx//2),groups=self.channels)
        return x[:,:,:,self.sx:-self.sx]


def replace_samples(model,mode):
    for name,child in list(model.named_children()):
        if isinstance(child,FixedSample):
            if mode=='half':
                child.__class__=HalfSample
            elif mode=='conv':
                source=int(max(child.y1.max(),child.x1.max()))+1
                height,width=child.y0.numel(),child.x0.numel()
                # Keep arbitrary crop and C32 mapping in its verified Gather form.
                probe=FixedSample(torch.tensor([[0.,0.,1.,1.]],device=child.yw.device),height,width,source)
                if source==64 and all(torch.equal(getattr(child,k),getattr(probe,k)) for k in ['y0','y1','x0','x1','yw','xw']):
                    # All complete-reference samples here use 12/16/24 projected channels.
                    channels=24 if hasattr(model,'reference') and getattr(model,'reference',None) is not None else None
                    if hasattr(model,'reference') and hasattr(model.reference,'project'):channels=model.reference.project.out_channels
                    if hasattr(model,'model') and hasattr(model.model,'reference'):channels=model.model.reference.project.out_channels
                    if hasattr(model,'projection_after'):channels=model.reference.project.in_channels if model.projection_after else model.reference.project.out_channels
                    if channels is not None:setattr(model,name,ConvSample(channels,height,width).to(child.yw.device))
        else:replace_samples(child,mode)
    return model


class NightFixed(nn.Module):
    def __init__(self,source,scene,fuse=False):
        super().__init__();self.source=copy.deepcopy(source);self.scene=scene
        self.sample=FixedSample(torch.tensor([[0.,0.,1.,1.]]),512,640)
        self.resizes=nn.ModuleList([FixedSample(torch.tensor([[0.,0.,1.,1.]]),64,64,64//s) for s in (2,4,8)])
        self.output=RepeatByteReorder(2)
        self.fused=LinearTail(source.core.model.tail,source.output.conv) if fuse else None

    def forward(self,raw,context):
        s=self.source;owner=s.core.model
        body=owner.body(owner.head(s.core.half_input(s.front(raw))))
        ref=owner.global_reference;c=s.core.half_input(context)
        value=ref.encoder(c)
        if hasattr(ref,'pyramid'):
            for i in getattr(ref,'keep',range(len(ref.pyramid))):
                value=value+self.resizes[i](ref.pyramid[i](c).float()).to(value.dtype)
        value=value+value.mean((-2,-1),keepdim=True)
        if self.scene=='ordinary':
            reference=ref.project(self.sample(value.float()).to(body.dtype))
            phases=self.fused(body+reference) if self.fused is not None else s.output.conv(owner.tail(body+reference))
        else:
            contribution=s.low_ref(ref.project(value))
            reference=self.sample(contribution.float()).to(body.dtype)
            phases=(self.fused(body) if self.fused is not None else s.output.conv(owner.tail(body)))+reference
        return self.output(phases.clamp(0,1)*255)


class LinearTail(nn.Module):
    def __init__(self,tail,projection):
        super().__init__()
        linear=tail[1].rep_conv if isinstance(tail,nn.Sequential) and isinstance(tail[0],nn.Identity) else tail
        assert isinstance(linear,nn.Conv2d) and linear.kernel_size==projection.kernel_size==(3,3)
        t=linear.weight.detach().cpu().double();p=projection.weight.detach().cpu().double()
        w=torch.zeros(p.shape[0],t.shape[1],5,5,dtype=torch.float64)
        for py in range(3):
            for px in range(3):
                for ty in range(3):
                    for tx in range(3):w[:,:,py+ty,px+tx]+=p[:,:,py,px]@t[:,:,ty,tx]
        bias=projection.bias.detach().cpu().double()+p.sum((-2,-1))@linear.bias.detach().cpu().double()
        self.conv=nn.Conv2d(t.shape[1],p.shape[0],5,padding=2,device=projection.weight.device,dtype=projection.weight.dtype)
        with torch.no_grad():self.conv.weight.copy_(w);self.conv.bias.copy_(bias)
        self.tail=copy.deepcopy(tail);self.projection=copy.deepcopy(projection)

    def forward(self,x):
        v=self.conv(x)
        top=self.projection(self.tail(x[:,:,:3,:]))[:,:,:1,:]
        bottom=self.projection(self.tail(x[:,:,-3:,:]))[:,:,-1:,:]
        left=self.projection(self.tail(x[:,:,:,:3]))[:,:,:,:1]
        right=self.projection(self.tail(x[:,:,:,-3:]))[:,:,:,-1:]
        mid=torch.cat((left[:,:,1:-1],v[:,:,1:-1,1:-1],right[:,:,1:-1]),-1)
        return torch.cat((top,mid,bottom),-2)
