"""Preserve signed native phase contrast without a learned input threshold."""
import torch
from torch import nn
from torch.nn import functional as F
from historydisplay_phase11 import CurrentRouteBoard,detail_tile

class PhaseBasis(nn.Module):
    def __init__(self,count):
        super().__init__();self.count=int(count);self.encode_folded=False
        self.encode=nn.Conv2d(4,16,4,stride=2,padding=1)
        self.spatial=nn.Conv2d(16,16,3,padding=1,groups=16,bias=False)
        self.context=nn.Conv2d(16,16,3,padding=3,dilation=3,groups=16,bias=False)
        self.mix=nn.Conv2d(16,16,1,bias=False)
        self.display_decode=nn.Conv2d(16,4,1)
        self.auxiliary=nn.Conv2d(16,8,1)
        with torch.no_grad():
            self.encode.weight.zero_();self.encode.bias.zero_()
            for phase in range(4):
                y,x=1+phase//2,1+phase%2
                for sign in range(2):
                    c=2*phase+sign;value=1 if sign==0 else -1
                    self.encode.weight[c,2].fill_(-value/16)
                    self.encode.weight[c,2,y,x]+=value
                    self.encode.weight[c+8,3,y,x]=-value*self.count/(self.count-1)
            for m in [self.spatial,self.context]:m.weight.zero_();m.weight[:,:,1,1]=1
            self.mix.weight.zero_()
            for c in range(16):self.mix.weight[c,c,0,0]=1
            self.display_decode.weight.zero_();self.display_decode.bias.zero_()
            self.auxiliary.weight.zero_();self.auxiliary.bias.fill_(-2)
        self.encode.bias.requires_grad_(False)
    def project(self):
        with torch.no_grad():
            w=self.encode.weight[:,:3];w.sub_(w.mean((1,2,3),keepdim=True))
            self.encode.bias.zero_()
    def fold_encoder(self):
        from mean_context_encoder11 import fold
        fold(self)
    def hidden(self,raw):
        raw=raw[:,-self.count:].to(self.encode.weight.dtype)
        if self.encode_folded:x=self.encode(raw)
        else:
            source=torch.cat([raw[:,-3:],raw.mean(1,keepdim=True)-raw[:,-1:]],1)
            x=F.conv2d(source,self.encode.weight*255,None,stride=2,padding=1)
        return F.relu(self.mix(self.context(self.spatial(F.relu(x)))))
    def evaluate(self,raw,auxiliary):
        raw=raw[:,-self.count:].half()
        def evaluate():
            latent=self.hidden(raw)
            return raw[:,-1:],self.display_decode(latent)*(1/255),self.auxiliary(latent) if auxiliary else None
        if self.encode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):return evaluate()
        return evaluate()
    def joint(self,raw):return self.evaluate(raw,True)
    def predict(self,raw):return self.evaluate(raw,False)[:2]
