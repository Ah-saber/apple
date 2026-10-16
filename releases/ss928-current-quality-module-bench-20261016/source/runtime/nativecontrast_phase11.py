"""Predict clean current RAW before reconstruction; no late detail route."""
import torch
from torch import nn
from torch.nn import functional as F

class RawDenoiser(nn.Module):
    def __init__(self):
        super().__init__();self.count=3;self.encode_folded=False
        self.encode=nn.Conv2d(3,16,4,stride=2,padding=1)
        self.spatial=nn.Conv2d(16,16,3,padding=1,groups=16)
        self.context=nn.Conv2d(16,16,3,padding=3,dilation=3,groups=16)
        self.mix=nn.Conv2d(16,8,1);self.decode=nn.Conv2d(8,4,1)
        self.auxiliary=nn.Conv2d(8,8,1);self.display_decode=nn.Conv2d(8,4,1)
        kernel=torch.zeros(4,1,2,2)
        for phase in range(4):kernel[phase,0,phase//2,phase%2]=1
        self.register_buffer('phase_kernel',kernel)
        with torch.no_grad():
            self.encode.weight.zero_();self.encode.bias.fill_(-.5)
            for phase in range(4):
                y,x=1+phase//2,1+phase%2
                for sign in range(2):
                    c=2*phase+sign;scale=1 if sign==0 else -1
                    self.encode.weight[c,-1].fill_(-scale/16);self.encode.weight[c,-1,y,x]+=scale
                    self.encode.weight[c+8,-1,y,x]=scale;self.encode.weight[c+8,:-1,y,x]=-scale/2
            for conv in [self.spatial,self.context]:
                conv.weight.zero_();conv.weight[:,:,1,1]=1;conv.bias.zero_()
            self.mix.weight.zero_();self.mix.bias.fill_(.1)
            for c in range(8):self.mix.weight[c,c,0,0]=1
            self.decode.weight.zero_();self.decode.bias.zero_()
            self.auxiliary.weight.zero_();self.auxiliary.bias.fill_(-2)
            self.display_decode.weight.zero_();self.display_decode.bias.zero_()
    def fold_encoder(self):
        if self.encode_folded:raise ValueError('Encoder already folded')
        with torch.no_grad():self.encode.weight.mul_(255)
        self.encode_folded=True
    def project(self):
        with torch.no_grad():self.encode.weight.sub_(self.encode.weight.mean((1,2,3),keepdim=True))
    def hidden(self,raw,base_phases=None):
        raw=raw[:,-3:].to(self.encode.weight.dtype)
        x=self.encode(raw) if self.encode_folded else F.conv2d(raw,self.encode.weight*255,self.encode.bias,stride=2,padding=1)
        return F.relu(self.mix(self.context(self.spatial(F.relu(x)))))
    def phases(self,raw):return self.decode(self.hidden(raw))*(1/255)
    def predict(self,raw):
        if self.encode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                latent=self.hidden(raw)
                phases=self.decode(latent)*(1/255)
                display=self.display_decode(latent)*(1/255)
                correction=F.conv_transpose2d(phases,self.phase_kernel.to(phases.dtype),stride=2)
        else:
            latent=self.hidden(raw)
            phases=self.decode(latent)*(1/255)
            display=self.display_decode(latent)*(1/255)
            correction=F.conv_transpose2d(phases,self.phase_kernel,stride=2)
        return (raw[:,-1:]+correction).to(torch.float16),display
    def forward(self,raw):return self.predict(raw)[0]

class CurrentRouteBoard(nn.Module):
    def __init__(self,reference):
        super().__init__();self.reference=reference
        self.branch=RawDenoiser().to(device=next(reference.parameters()).device,dtype=torch.float32)
    @property
    def kind(self):return self.reference.kind
    @property
    def source_count(self):return 3
    @property
    def board(self):return self.reference.board
    @property
    def accepted(self):return self.reference.accepted
    def forward(self,raw,context):
        clean,display=self.branch.predict(raw[:,-3:])
        base=self.reference.phases(clean,context)
        phases=(base+display.to(base.dtype)).to(base.dtype)
        output=self.board.output if self.kind else self.board.rows
        return output(phases.clamp(0,1)*255)

def detail_tile(model,features,field,prior,raw):
    from compact_student import tile_forward,front_feature
    from training_prior9 import input_prior
    clean,display=model.branch.predict(raw)
    features=front_feature(model.reference,clean).float()
    pred,body=tile_forward(model.reference,features,field,input_prior(model,clean))
    return pred+F.pixel_shuffle(display,2),body,F.pixel_unshuffle(pred,2)
