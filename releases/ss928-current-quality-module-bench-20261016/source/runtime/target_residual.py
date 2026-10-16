"""Small phase residual with zero response to identical temporal inputs."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from train_guided import setup
from run_verified import half_board
from direct_motion import DirectBoard

ROOT=Path(__file__).resolve().parents[1]

def current_setup(scene):
    kind,_,config,df,box_for,_,count,_,base,original,_=setup(scene)
    if kind=='special': base,count=copy.deepcopy(original),9
    family=next((p for p in ['budget_profile_','combined_front_','cohort_stream_'] if (ROOT/(p+scene)/'summary.json').exists()),None)
    profile=json.loads((ROOT/(family+scene)/'summary.json').read_text())['pipeline'] if family else 'control9'
    if family=='combined_front_' or (family=='budget_profile_' and scene!='weather_light'):
        saved=torch.load(ROOT/('multiscale_front_'+scene)/'best.pt',map_location='cpu',weights_only=False)
        (base.source.front if kind else base.quarter.front).load_state_dict(saved['front'])
    return kind,config,df,box_for,count,base,original,profile

class PhaseResidual(nn.Module):
    def __init__(self,count):
        super().__init__()
        self.conv=nn.Conv2d(count,4,2,stride=2,bias=False)
        nn.init.zeros_(self.conv.weight)

    def project(self):
        with torch.no_grad():
            self.conv.weight.sub_(self.conv.weight.mean(1,keepdim=True))
            self.conv.weight.clamp_(-.8,.8)
            self.conv.weight.sub_(self.conv.weight.mean(1,keepdim=True))

    def forward(self,raw): return self.conv(raw.to(self.conv.weight.dtype))

class ShrinkResidual(nn.Module):
    """Suppress small residuals with Conv, ReLU and a fixed signed projection."""
    def __init__(self,count,weights,threshold_gray):
        super().__init__()
        self.count=count
        size=weights['conv.weight'].shape[-1]
        self.conv=nn.Conv2d(count,8,size,stride=2,padding=(size-2)//2,bias=True)
        self.combine=nn.Conv2d(8,4,1,bias=False)
        with torch.no_grad():
            w=weights['conv.weight']
            self.conv.weight.copy_(torch.cat([w,-w],0))
            self.conv.bias.fill_(-threshold_gray/255)
            self.combine.weight.zero_()
            for i in range(4):
                self.combine.weight[i,i,0,0]=1
                self.combine.weight[i,i+4,0,0]=-1

    def forward(self,raw):
        return self.combine(F.relu(self.conv(raw.to(self.conv.weight.dtype))))

class SpatialResidual(PhaseResidual):
    """Use neighboring temporal differences to reduce a pixel's independent noise."""
    def __init__(self,count):
        nn.Module.__init__(self)
        self.conv=nn.Conv2d(count,4,4,stride=2,padding=1,bias=False)
        nn.init.zeros_(self.conv.weight)

class ResidualBoard(DirectBoard):
    def __init__(self,board,branch,count,kind):
        super().__init__(board,branch,count,kind)

    def phases(self,raw,context):
        return self.night_phases(raw,context) if self.kind else self.board.stages(raw,context)['preclip_phases']

    def forward(self,raw,context):
        phases=self.phases(raw,context)+self.branch(raw)
        output=self.board.output if self.kind else self.board.rows
        return output(phases.clamp(0,1)*255)

def make_board(base,branch,kind,box):
    current=base if kind else half_board(base,box.cuda())
    return ResidualBoard(current,branch,branch.conv.in_channels,kind).cuda().eval()
