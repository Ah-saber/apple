"""Restore current bodies using foreground features over a 33-pixel context."""
import torch
from torch import nn
from residual50 import local
from road_residual53 import support53
from body_gain76 import history_fields76

class BodyTemporal81(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(16,32,3,padding=1,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=2,dilation=2,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=8,dilation=8,padding_mode='reflect'),nn.ReLU(),
            nn.Conv2d(32,2,3,padding=1,padding_mode='reflect'))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor([0.,-3.]))
    def forward(self,v,source,previous_source):
        raw=v['raw50'];history=v['history76']
        x=torch.cat([raw[:,:8],raw[:,12:13],history[:,1:4],v['presence76']/16,
                     v['instant_presence76']/16,v['aligned78']/6,local(source-previous_source,21)/16],1)
        heads=self.net(x.clamp(-4,4));signed_body=v['signed_gate80']
        fore=history[:,1:2]*16;excess=(history[:,2:3]-history[:,1:2])*16
        amplitude=16*torch.tanh(heads[:,0:1])
        amplitude=torch.maximum(amplitude,-torch.minimum(.5*fore.abs(),torch.full_like(fore,4.)))
        motion=raw[:,12:13]*16;instant=raw[:,:8].mean(1,keepdim=True)*16
        raw_noise,sr_noise=v['noise76'][0],v['noise76'][1]
        departure=((excess*motion<0)&(instant*motion>0)).float()*((excess.abs()-sr_noise)/sr_noise).clamp(0,1)
        departure*=((motion.abs()-.5*raw_noise)/raw_noise).clamp(0,1)*v['coherent76']
        correction=torch.where(signed_body.abs()>0,amplitude*signed_body,8*torch.sigmoid(heads[:,1:2])*departure*motion.sign())
        return correction*support53(v['crop53'],source.device)
