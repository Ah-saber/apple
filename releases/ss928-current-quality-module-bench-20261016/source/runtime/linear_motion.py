"""Causal linear residual with a fused phase convolution."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from direct_motion import DirectBoard as NonlinearBoard

class LinearMotion(nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.count=count
        kernel=torch.zeros(2,count,1,1)
        kernel[0,-1]=1
        kernel[0,:-1]=-1/(count-1)
        kernel[1,-1]=1
        kernel[1,-2]=-1
        self.register_buffer('delta_kernel',kernel)
        self.mean_gain=nn.Parameter(torch.zeros(()))
        self.previous_gain=nn.Parameter(torch.zeros(()))
        self.offset=nn.Parameter(torch.zeros(()), requires_grad=False)

    def calibrate(self,samples):
        return

    def project(self):
        with torch.no_grad():
            self.mean_gain.clamp_(-1,1)
            self.previous_gain.clamp_(-1,1)
            self.offset.clamp_(-.04,.04)

    def forward(self,raw):
        d=F.conv2d(raw[:,-self.count:].to(self.delta_kernel.dtype),self.delta_kernel)
        return self.mean_gain*d[:,0:1]+self.previous_gain*d[:,1:2]+self.offset

class LinearBoard(NonlinearBoard):
    def __init__(self,board,branch,source_count,kind=None):
        super().__init__(board,branch,source_count,kind)
        count=branch.count
        coeff=(branch.mean_gain.detach()*branch.delta_kernel[0,:,0,0]+
               branch.previous_gain.detach()*branch.delta_kernel[1,:,0,0]).float()
        if kind is None and self.board.base.quarter.raw_skip:
            coeff[-1]+=1
            self.board.base.quarter.raw_skip=False
        kernel=torch.zeros(4,count,2,2,device=coeff.device,dtype=torch.float32)
        for phase in range(4):
            kernel[phase,:,phase//2,phase%2]=coeff
        self.register_buffer('fused_kernel',kernel.half())
        self.register_buffer('fused_bias',branch.offset.detach().expand(4).clone().half())

    def forward(self,raw,context):
        source_raw=raw[:,-self.source_count:]
        phases=(self.night_phases(source_raw,context) if self.kind else
                self.board.stages(source_raw,context)['preclip_phases'])
        residual=F.conv2d(raw[:,-self.branch.count:].to(self.fused_kernel.dtype),
                          self.fused_kernel,self.fused_bias,stride=2)
        output=self.board.output if self.kind else self.board.rows
        return output((phases+residual).clamp(0,1)*255)

