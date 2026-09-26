"""Complete model I/O; alternative NPU graph expressions, no ARM partition."""
import copy, types
import torch
from torch import nn
from torch.nn import functional as F


class DepthToSpaceDCR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # With one output channel, DCR and CRD have the same permutation.
        if x.shape[1] != 36:raise ValueError('DCR equivalence requires single-channel output')
        return F.pixel_shuffle(x,6)
    @staticmethod
    def symbolic(g,x):
        return g.op('DepthToSpace',x,blocksize_i=6,mode_s='DCR')


def phase_permutation(first,second):
    if first*second!=6:raise ValueError('Expected total factor six')
    order=[None]*36
    for ia in range(first):
        for ja in range(first):
            for ib in range(second):
                for jb in range(second):
                    new=(ib*second+jb)*first*first+ia*first+ja
                    old=6*(second*ia+ib)+(second*ja+jb)
                    order[new]=old
    return order


class FullGraphNine(nn.Module):
    def __init__(self,core,layout='crd6',output_precision='float32'):
        super().__init__()
        if layout not in ('crd6','dcr6','staged23','staged32','reshape','native_deconv'):
            raise ValueError('Unknown layout')
        if output_precision not in ('float32','float16'):raise ValueError('Unknown output precision')
        self.core=copy.deepcopy(core)
        self.layout=layout;self.output_precision=output_precision
        if layout in ('staged23','staged32'):
            first,second=(2,3) if layout=='staged23' else (3,2)
            order=phase_permutation(first,second)
            conv=self.core.model.upsample[0]
            with torch.no_grad():
                conv.weight.copy_(conv.weight[order].clone())
                conv.bias.copy_(conv.bias[order].clone())
        if layout=='native_deconv':
            self.deconvolution=NativeOutput(self.core.model.upsample[0])

    def forward(self,stack,context):
        c,m=self.core,self.core.model
        current=stack[:,-1:]
        gate_input=torch.cat((stack,m._trajectory_features(stack)),1)
        joined=c.second(F.relu(c.first(c.half_input(gate_input)))).float()
        correction,gate=joined[:,:1],torch.sigmoid(joined[:,1:])
        denoised=current.float()+correction*(1.-gate)
        features=m.body(m.head(m.down(c.half_input(denoised))))
        reference=m.global_reference.encode(c.half_input(context))
        reference=m.global_reference.project(reference+reference.mean((-2,-1),keepdim=True))
        reference=F.interpolate(reference,size=features.shape[-2:],mode='bilinear',align_corners=False)
        tail=m.tail(features+reference)
        if self.layout=='native_deconv':
            gray=self.deconvolution(tail).float().clamp(0,1)*255
        else:
            packed=m.upsample[0](tail).float().clamp(0,1)*255
            if self.layout=='dcr6':gray=DepthToSpaceDCR.apply(packed)
            elif self.layout=='staged23':gray=F.pixel_shuffle(F.pixel_shuffle(packed,2),3)
            elif self.layout=='staged32':gray=F.pixel_shuffle(F.pixel_shuffle(packed,3),2)
            elif self.layout=='reshape':
                n,_,h,w=packed.shape
                gray=packed.reshape(n,1,6,6,h,w).permute(0,1,4,2,5,3).reshape(n,1,h*6,w*6)
            else:gray=F.pixel_shuffle(packed,6)
        return gray.half() if self.output_precision=='float16' else gray


class NativeOutput(nn.Module):
    """Conv3x3+CRD6 as a stride6 ConvTranspose18; phase bias included in weight."""
    def __init__(self,conv):
        super().__init__()
        if (tuple(conv.kernel_size)!=(3,3) or conv.out_channels!=36 or conv.groups!=1
                or tuple(conv.stride)!=(1,1) or tuple(conv.padding)!=(1,1)
                or tuple(conv.dilation)!=(1,1)):
            raise ValueError('Unexpected frozen output layer')
        channels=conv.in_channels
        self.layer=nn.ConvTranspose2d(channels+1,1,18,stride=6,padding=6,bias=False,
                                      device=conv.weight.device,dtype=conv.weight.dtype)
        with torch.no_grad():
            self.layer.weight.zero_()
            for i in range(6):
                for j in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            self.layer.weight[:channels,0,(2-ky)*6+i,(2-kx)*6+j]=conv.weight[6*i+j,:,ky,kx]
                    self.layer.weight[channels,0,6+i,6+j]=conv.bias[6*i+j]
        self.layer.to(memory_format=torch.channels_last)
    def forward(self,x):
        return self.layer(torch.cat((x,torch.ones_like(x[:,:1])),1))
