"""Expose nine physical RAW observations without discarding weak differences."""
import gc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from target_residual import current_setup
from raw_gain23 import RawGain23
from repair_motion15 import features

class RawFeatures24:
    def __init__(self,scene):
        kind,config,df,box,count,base,original,profile=current_setup(scene)
        self.root=df(config,'train').base.root
        del base,original;gc.collect();torch.cuda.empty_cache()
        self.open={};self.scale={}
    def for_record(self,row,frame,crop):
        path=row['raw']['path']
        if path not in self.open:
            self.open[path]=np.load(self.root/path,mmap_mode='r')
            q=np.quantile(self.open[path][0],[.01,.99]);self.scale[path]=max(float(q[1]-q[0]),1.)
        y,x,h,w=crop;ids=np.maximum(np.arange(frame-8,frame+1),0)
        a=np.array(self.open[path][ids,y:y+h,x:x+w],dtype=np.float32)*(255/self.scale[path])
        raw=torch.from_numpy(a)[None].cuda()
        current=raw[:,-1:]
        difference=(current-raw[:,:-1])/16
        local=(current-F.avg_pool2d(current,9,1,4,count_include_pad=False))/16
        return torch.cat([difference,local],1).clamp(-4,4)

class RawTemporal24(RawGain23):
    def __init__(self,parent=None):
        super().__init__(parent)
        old=self.body[0]
        expanded=nn.Conv2d(19,24,3,padding=1)
        with torch.no_grad():
            expanded.weight[:,:10].copy_(old.weight);expanded.weight[:,10:].zero_();expanded.bias.copy_(old.bias)
        expanded.bias.requires_grad_(False)
        # Preserve the existing image feature path; only the new RAW channels learn.
        expanded.weight.register_hook(lambda gradient:torch.cat([torch.zeros_like(gradient[:,:10]),gradient[:,10:]],1))
        self.body[0]=expanded
    def forward(self,v,previous,previous_output,initial=False):
        joined=torch.cat([features(v,previous),v['raw24']],1)
        latent=self.body(joined)
        fixed=v['baseline']+16*torch.tanh(self.correct(latent))
        heads=self.gate(latent)
        corrected=fixed+torch.tanh(heads[:,2:3])*(v['current_only']-v['baseline'])
        corrected=corrected+torch.tanh(heads[:,3:4])*(v['current_only']-v['past_background'])
        motion=torch.sigmoid(heads[:,1:2]);carry=.95*torch.sigmoid(heads[:,0:1])*(1-motion)
        return corrected if initial else corrected*(1-carry)+previous_output*carry,motion,carry,corrected
