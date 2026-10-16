"""Separate signed current-body correction from departed-history correction."""
import torch
from torch import nn
from residual50 import local
from road_residual53 import support53
from body_gain76 import history_fields76

class BodyGain80(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(28,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.ReLU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor([0.,-3.,0.]))
    def forward(self,v,source,previous_source):
        current=v['current_only'];raw=v['raw50'];history=v['history76']
        x=torch.cat([v['baseline']/255,source/255,local(source)/16,(current-source)/16,
                     local(current)/16,(current-v['past_background'])/16,v['activity'],v['occupancy'],
                     local(source-previous_source,21)/16,raw,history,v['presence76']/16,v['aligned78']/6],1)
        heads=self.net(x.clamp(-4,4))
        raw_noise=v['noise76'][0];sr_noise=v['noise76'][1]
        presence=v['presence76'];current_presence=v['instant_presence76']
        motion=raw[:,12:13]*16;instant=raw[:,:8].mean(1,keepdim=True)*16
        fore=history[:,1:2]*16;excess=(history[:,2:3]-history[:,1:2])*16
        coherent=v['coherent76']
        signed_body=v['signed_gate80'];body=signed_body.abs()
        body_amplitude=8*torch.tanh(heads[:,0:1])+8*torch.tanh(heads[:,2:3])
        body_amplitude=torch.maximum(body_amplitude,-torch.minimum(.5*fore.abs(),torch.full_like(fore,4.)))
        departure=((excess*motion<0)&(instant*motion>0)).float()*((excess.abs()-sr_noise)/sr_noise).clamp(0,1)
        departure*=((motion.abs()-.5*raw_noise)/raw_noise).clamp(0,1)*coherent
        departure_amplitude=8*torch.sigmoid(heads[:,1:2])
        correction=torch.where(body>0,body_amplitude*signed_body,departure_amplitude*departure*motion.sign())
        return correction*support53(v['crop53'],source.device)


