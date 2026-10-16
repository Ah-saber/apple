"""Learn spatial motion features before thresholding; two inference convolutions."""
import torch
from torch import nn
from torch.nn import functional as F
from timematch_phase11 import TimeMatch,CurrentRouteBoard

class SpatialTime(nn.Module):
    def __init__(self,count):
        super().__init__();self.count=int(count)
        self.encode=nn.Conv2d(4*(count-1),8,3,padding=0,bias=True)
        self.decode=nn.Conv2d(8,4,1,bias=False)
        self.auxiliary=nn.Conv2d(8,8,1)
        with torch.no_grad():
            self.encode.weight.zero_();self.encode.bias.fill_(-.5)
            for phase in range(4):
                for frame in range(count-1):
                    self.encode.weight[phase,4*frame+phase,1,1]=1/(count-1)
                    self.encode.weight[phase+4,4*frame+phase,1,1]=-1/(count-1)
            self.decode.weight.zero_();self.auxiliary.weight.zero_();self.auxiliary.bias.zero_()
        self.project()
    def features(self,raw):
        raw=raw[:,-self.count:].half();current=TimeMatch.phase(raw[:,-1:])
        difference=torch.cat([current-TimeMatch.phase(raw[:,i:i+1]) for i in range(self.count-1)],1)*255
        return F.relu(self.encode(TimeMatch.copy_edge(difference)))
    def project(self):
        with torch.no_grad():
            self.encode.weight.sub_(self.encode.weight.mean((2,3),keepdim=True))
            self.encode.bias.clamp_(max=0)
    def evaluate(self,raw,auxiliary=False):
        def forward():
            shared=self.features(raw)
            return self.decode(shared),self.auxiliary(shared).float() if auxiliary else None
        if self.decode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):return forward()
        return forward()
    def predict(self,raw):
        delta,_=self.evaluate(raw);return raw[:,-1:],delta*(1/255)

