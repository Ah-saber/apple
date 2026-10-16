"""Learn bounded departure correction with past32 independent SR context."""
import numpy as np
import cv2
import torch
from torch import nn
from residual50 import local
from road_residual53 import support53

def history_fields71(current,i,crop,source,device='cuda',repair_boundary=False):
    y,x,h,w=crop
    ids=np.maximum(np.arange(i-32,i),0)
    histories=np.array(current[ids,y:y+h,x:x+w],dtype=np.float32)
    if repair_boundary and x==0:
        histories[:,:,:3]=histories[:,:,3:6][:,:,::-1]
    background=np.median(histories,axis=0)
    observation=np.array(current[i,y:y+h,x:x+w],dtype=np.float32)
    if repair_boundary and x==0:
        observation[:,:3]=observation[:,3:6][:,::-1]
    def hp(a,size):return a-cv2.blur(a,(size,size),borderType=cv2.BORDER_REFLECT)
    differences=np.stack([hp(a-background,21) for a in histories[-8:]])
    peak=np.argmax(np.abs(differences),axis=0)[None]
    signed=np.take_along_axis(differences,peak,axis=0)[0]
    fields=np.stack([hp(background,9),hp(observation-background,21),hp(source-background,21),signed])/16
    return torch.from_numpy(fields)[None].to(device).clamp(-4,4)

class HistoryResidual71(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(26,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.ReLU(),nn.Conv2d(32,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,v,source,previous_source):
        current=v['current_only']
        x=torch.cat([v['baseline']/255,source/255,local(source)/16,(current-source)/16,
                     local(current)/16,(current-v['past_background'])/16,v['activity'],v['occupancy'],
                     local(source-previous_source,21)/16,v['raw50'],v['history71']],1)
        delta=4*torch.tanh(self.net(x.clamp(-4,4)))
        return delta*support53(v.get('crop53',(0,0,*source.shape[-2:])),source.device)
