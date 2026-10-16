"""RAW-guided motion and local contrast correction; standard phase convolutions."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from direct_motion import DirectMotion,DirectBoard

class GuidedMotion(DirectMotion):
    def __init__(self,count=5):
        super().__init__(count)
        self.raw_detail=nn.Parameter(torch.zeros(()))
        self.pred_detail=nn.Parameter(torch.zeros(()))
        self.offset.requires_grad_(False)
        self.noise_weight.data.fill_(1.5)

    def calibrate(self,samples):
        super().calibrate(samples)
        self.threshold.data.copy_(1/self.inverse_scale)

    def project(self):
        with torch.no_grad():
            self.mean_gain.clamp_(-2,2);self.previous_gain.clamp_(-2,2)
            self.raw_detail.clamp_(-8,8);self.pred_detail.clamp_(-1.5,1.5)
            self.noise_weight.clamp_(0,4);self.threshold.clamp_(0,.25)

    def forward(self,raw,baseline):
        d=self.differences(raw)
        signal=torch.maximum(d[:,0:1].abs(),d[:,1:2].abs())
        peak=F.max_pool2d(signal,2,2)
        background=F.avg_pool2d(peak,3,1,1)
        gate=((F.max_pool2d(peak,3,1,1)-self.noise_weight*background-self.threshold)*self.inverse_scale).clamp(0,1)
        gate=gate.repeat_interleave(2,-2).repeat_interleave(2,-1)
        current=raw[:,-1:].to(d.dtype)
        raw_detail=current-F.avg_pool2d(current,3,1,1)
        pred_detail=baseline-F.avg_pool2d(baseline,3,1,1)
        return gate*(self.mean_gain*d[:,0:1]+self.previous_gain*d[:,1:2]+
                     self.raw_detail*raw_detail+self.pred_detail*pred_detail)

class PhaseGuided(nn.Module):
    def __init__(self,branch):
        super().__init__()
        self.branch=copy.deepcopy(branch).half().eval()
        count=branch.count
        kernel=torch.zeros(12,count,2,2,device=branch.delta_kernel.device)
        for phase in range(4):
            kernel[phase,:,phase//2,phase%2]=branch.delta_kernel[0,:,0,0]
            kernel[phase+4,:,phase//2,phase%2]=branch.delta_kernel[1,:,0,0]
            kernel[phase+8,-1,phase//2,phase%2]=1
        mean=torch.zeros(4,4,3,3,device=kernel.device)
        for phase in range(4):
            py,px=phase//2,phase%2
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    sy,sx=py+dy,px+dx
                    mean[phase,(sy%2)*2+sx%2,1+sy//2,1+sx//2]+=1/9
        self.register_buffer('input_kernel',kernel.half())
        self.register_buffer('mean_kernel',mean.half())
        self.register_buffer('gate_copy',torch.ones(4,1,1,1,device=kernel.device,dtype=torch.float16))

    def forward(self,raw,phases):
        values=F.conv2d(raw[:,-self.branch.count:].to(self.input_kernel.dtype),self.input_kernel,stride=2)
        mean,previous,current=values[:,:4],values[:,4:8],values[:,8:]
        signal=torch.maximum(mean.abs(),previous.abs())
        peak=torch.maximum(torch.maximum(signal[:,0:1],signal[:,1:2]),torch.maximum(signal[:,2:3],signal[:,3:4]))
        background=F.avg_pool2d(peak,3,1,1)
        b=self.branch
        gate=((F.max_pool2d(peak,3,1,1)-b.noise_weight*background-b.threshold)*b.inverse_scale).clamp(0,1)
        gate=F.conv2d(gate,self.gate_copy)
        raw_detail=current-F.conv2d(current,self.mean_kernel,padding=1)
        pred_detail=phases-F.conv2d(phases,self.mean_kernel,padding=1)
        return gate*(b.mean_gain*mean+b.previous_gain*previous+b.raw_detail*raw_detail+b.pred_detail*pred_detail)

class GuidedBoard(DirectBoard):
    def __init__(self,board,branch,source_count,kind=None):
        super().__init__(board,branch,source_count,kind)
        self.guided=PhaseGuided(branch)
    def forward(self,raw,context):
        source_raw=raw[:,-self.source_count:]
        phases=(self.night_phases(source_raw,context) if self.kind else
                self.board.stages(source_raw,context)['preclip_phases'])
        correction=self.guided(raw,phases)
        output=self.board.output if self.kind else self.board.rows
        return output((phases+correction).clamp(0,1)*255)

