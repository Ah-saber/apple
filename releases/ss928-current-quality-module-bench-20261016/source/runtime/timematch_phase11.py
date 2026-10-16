"""One spatial motion correction, with explicit signed causal RAW differences."""
import torch
from torch import nn
from torch.nn import functional as F
from historydisplay_phase11 import CurrentRouteBoard

class TimeMatch(nn.Module):
    def __init__(self,count):
        super().__init__();self.count=int(count)
        self.decode=nn.Conv2d(8*(count-1),4,3,padding=0,bias=False)
        self.auxiliary=nn.Conv2d(4,8,1)
        with torch.no_grad():
            self.decode.weight.zero_();self.auxiliary.weight.zero_();self.auxiliary.bias.zero_()
    @staticmethod
    def phase(raw):
        return torch.cat([raw[...,::2,::2],raw[...,::2,1::2],raw[...,1::2,::2],raw[...,1::2,1::2]],1)
    def features(self,raw):
        raw=raw[:,-self.count:].half();current=self.phase(raw[:,-1:])
        delta=torch.cat([current-self.phase(raw[:,i:i+1]) for i in range(self.count-1)],1)*255
        delta=delta-F.avg_pool2d(delta,3,1,1,count_include_pad=False)
        return torch.cat([F.relu(delta),F.relu(-delta)],1)
    @staticmethod
    def copy_edge(value):
        value=torch.cat([value[...,:1,:],value,value[...,-1:,:]],dim=-2)
        return torch.cat([value[...,:,:1],value,value[...,:,-1:]],dim=-1)
    def project(self):
        with torch.no_grad():
            self.decode.weight.sub_(self.decode.weight.mean((2,3),keepdim=True))
    def evaluate(self,raw,auxiliary=False):
        def evaluate():
            correction=self.decode(self.copy_edge(self.features(raw)))
            return correction,self.auxiliary(correction).float() if auxiliary else None
        if self.decode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):return evaluate()
        return evaluate()
    def predict(self,raw):
        delta,_=self.evaluate(raw)
        return raw[:,-1:],delta*(1/255)

