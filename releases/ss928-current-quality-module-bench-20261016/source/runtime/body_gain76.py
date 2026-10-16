"""Learn RAW-signed body and departure amplitudes with available past64 context."""
import numpy as np
import cv2
import torch
from torch import nn
from residual50 import local
from road_residual53 import support53

def history_fields76(current,i,crop,source,device='cuda',repair_boundary=False):
    y,x,h,w=crop
    ids=np.arange(max(0,i-64),i) if i else np.array([0])
    histories=np.array(current[ids,y:y+h,x:x+w],dtype=np.float32)
    observation=np.array(current[i,y:y+h,x:x+w],dtype=np.float32)
    if repair_boundary and x==0:
        histories[:,:,:3]=histories[:,:,3:6][:,:,::-1];observation[:,:3]=observation[:,3:6][:,::-1]
    background=np.median(histories,axis=0)
    def hp(a,size):return a-cv2.blur(a,(size,size),borderType=cv2.BORDER_REFLECT)
    differences=np.stack([hp(a-background,21) for a in histories[-8:]])
    peak=np.argmax(np.abs(differences),axis=0)[None]
    signed=np.take_along_axis(differences,peak,axis=0)[0]
    fields=np.stack([hp(background,9),hp(observation-background,21),hp(source-background,21),signed])/16
    return torch.from_numpy(fields)[None].to(device).clamp(-4,4)

class BodyGain76(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(27,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.ReLU(),nn.Conv2d(32,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-3.)
    def forward(self,v,source,previous_source):
        current=v['current_only'];raw=v['raw50'];history=v['history76']
        x=torch.cat([v['baseline']/255,source/255,local(source)/16,(current-source)/16,
                     local(current)/16,(current-v['past_background'])/16,v['activity'],v['occupancy'],
                     local(source-previous_source,21)/16,raw,history,v['presence76']/16],1)
        amplitude=16*torch.sigmoid(self.net(x.clamp(-4,4)))
        raw_noise=v['noise76'][0];sr_noise=v['noise76'][1]
        presence=v['presence76'];current_presence=v['instant_presence76']
        motion=raw[:,12:13]*16;instant=raw[:,:8].mean(1,keepdim=True)*16
        fore=history[:,1:2]*16;excess=(history[:,2:3]-history[:,1:2])*16
        coherent=v['coherent76']
        temporal=((fore*presence>0)&(current_presence*presence>0)).float()
        temporal*=((fore.abs()-.5*sr_noise)/sr_noise).clamp(0,1)
        temporal*=torch.maximum(((presence.abs()-.5*raw_noise)/raw_noise).clamp(0,1),torch.full_like(presence,.25))*coherent
        spatial_raw=raw[:,8:9]*16;spatial_sr=local(current,9)
        spatial=((spatial_raw*spatial_sr>0).float()*((spatial_raw.abs()-2*raw_noise)/raw_noise).clamp(0,1)*((spatial_sr.abs()-2*sr_noise)/sr_noise).clamp(0,1)*coherent)
        body=torch.where(temporal>0,temporal,spatial)
        direction=torch.where(temporal>0,presence.sign(),spatial_raw.sign())
        departure=((excess*motion<0)&(instant*motion>0)).float()*((excess.abs()-sr_noise)/sr_noise).clamp(0,1)
        departure*=((motion.abs()-.5*raw_noise)/raw_noise).clamp(0,1)*coherent
        signed_gate=torch.where(body>0,body*direction,departure*motion.sign())
        return amplitude*signed_gate*support53(v['crop53'],source.device)
