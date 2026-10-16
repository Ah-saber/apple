"""Retain background history and add an ungated current-motion display head."""
import torch
from torch import nn
from torch.nn import functional as F
from meancontext_phase11 import RawDenoiser
from static_scatter11 import scatter
from gray_raw_phase_features11 import features

class HistoryFeatures(RawDenoiser):
    def joint(self,raw):
        raw=raw[:,-self.count:].half()
        def evaluate():
            latent=self.hidden(raw)
            clean=(raw[:,-1:]+scatter(self.decode(latent)*(1/255))).half()
            display=self.display_decode(torch.cat([latent,features(self,raw[:,-1:],clean)],1))*(1/255)
            return clean,display,self.auxiliary(latent)
        if self.encode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):return evaluate()
        return evaluate()

class CurrentRouteBoard(nn.Module):
    def __init__(self,reference,branch,count):
        super().__init__();self.reference=reference;self.branch=branch;self.count=count
    @property
    def source_count(self):return self.count
    @property
    def kind(self):return self.reference.kind
    @property
    def board(self):return self.reference.board
    @property
    def accepted(self):return self.reference.accepted
    def modified(self,raw):return raw[:,-self.count:].half()
    def parts(self,raw):
        raw=self.modified(raw);_,display,auxiliary=self.branch.joint(raw)
        return raw,display,auxiliary
    def forward(self,raw,context):
        raw=self.modified(raw);_,display=self.branch.predict(raw)
        phases=(self.reference.phases(raw,context.half())+display).half()
        output=self.board.output if self.kind else self.board.rows
        return output(phases.clamp(0,1)*255)

def detail_tile(model,features,field,prior,raw):
    from compact_student import tile_forward,front_feature
    from training_prior9 import input_prior
    raw=model.modified(raw);_,display=model.branch.predict(raw)
    pred,body=tile_forward(model.reference,front_feature(model.reference,raw).float(),field,input_prior(model,raw))
    return pred+F.pixel_shuffle(display,2),body,F.pixel_unshuffle(pred,2)
