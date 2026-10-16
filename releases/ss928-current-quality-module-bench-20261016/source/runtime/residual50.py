"""Current C32 detail residual on top of the preserved stable output."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from raw_temporal24 import RawFeatures24

def local(a,size=9):
    return a-F.avg_pool2d(a,size,1,size//2,count_include_pad=False)

class RawSignal50(RawFeatures24):
    def for_record(self,row,frame,crop):
        path=row['raw']['path']
        if path not in self.open:
            self.open[path]=np.load(self.root/path,mmap_mode='r')
            q=np.quantile(self.open[path][0],[.01,.99]);self.scale[path]=max(float(q[1]-q[0]),1.)
        y,x,h,w=crop;ids=np.maximum(np.arange(frame-8,frame+1),0)
        a=torch.from_numpy(np.array(self.open[path][ids,y:y+h,x:x+w],dtype=np.float32))[None].cuda()*(255/self.scale[path])
        current=a[:,-1:]
        differences=local((current-a[:,:-1])/16,21)
        means=[a[:,-n:].mean(1,keepdim=True) for n in [3,5,9]]
        planes=[local(current)/16]+[local(m)/16 for m in means]+[local(means[0]-means[-1],21)/16]
        return torch.cat([differences]+planes,1).clamp(-4,4)

class Residual50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(22,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,v,source,previous_source):
        current=v['current_only']
        x=torch.cat([v['baseline']/255,source/255,local(source)/16,(current-source)/16,
            local(current)/16,(current-v['past_background'])/16,v['activity'],v['occupancy'],
            local(source-previous_source,21)/16,v['raw50']],1)
        delta=8*torch.tanh(self.net(x.clamp(-4,4)))
        return delta
