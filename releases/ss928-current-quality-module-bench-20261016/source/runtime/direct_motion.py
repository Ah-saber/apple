"""Signed causal RAW innovation applied at native output pixels."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class DirectMotion(nn.Module):
    def __init__(self, count=5):
        super().__init__()
        self.count = count
        kernel = torch.zeros(2, count, 1, 1)
        kernel[0, -1] = 1
        kernel[0, :-1] = -1/(count-1)
        kernel[1, -1] = 1
        kernel[1, -2] = -1
        self.register_buffer('delta_kernel', kernel)
        self.register_buffer('inverse_scale', torch.ones(()))
        self.mean_gain = nn.Parameter(torch.zeros(()))
        self.previous_gain = nn.Parameter(torch.zeros(()))
        self.noise_weight = nn.Parameter(torch.ones(()))
        self.threshold = nn.Parameter(torch.zeros(()))
        self.offset = nn.Parameter(torch.zeros(()))

    def differences(self, raw):
        return F.conv2d(raw[:, -self.count:].to(self.delta_kernel.dtype), self.delta_kernel)

    def calibrate(self, samples):
        with torch.no_grad():
            scales = []
            for raw in samples:
                d = self.differences(raw)
                signal = torch.maximum(d[:, 0:1].abs(), d[:, 1:2].abs())
                scales.append(torch.quantile(signal.float().flatten(), .90).clamp_min(1e-4))
            self.inverse_scale.copy_(1/torch.stack(scales).mean())

    def project(self):
        with torch.no_grad():
            self.mean_gain.clamp_(-.5, .5)
            self.previous_gain.clamp_(-.5, .5)
            self.noise_weight.clamp_(0, 2)
            self.threshold.clamp_(0, .25)
            self.offset.clamp_(-.04, .04)

    def forward(self, raw):
        d = self.differences(raw)
        signal = torch.maximum(d[:, 0:1].abs(), d[:, 1:2].abs())
        background = F.avg_pool2d(signal, 3, 1, 1, count_include_pad=False)
        normalized = ((signal-self.noise_weight*background-self.threshold)*self.inverse_scale).clamp(-128, 128)
        gate = normalized.clamp(0, 1)
        return gate*(self.mean_gain*d[:, 0:1]+self.previous_gain*d[:, 1:2])+self.offset


class DirectBoard(nn.Module):
    def __init__(self, board, branch, source_count, kind=None):
        super().__init__()
        self.board = copy.deepcopy(board)
        self.branch = copy.deepcopy(branch).half().eval()
        self.source_count, self.kind = source_count, kind
        phases = torch.zeros(4, 1, 2, 2, device=next(board.parameters()).device, dtype=torch.float16)
        for i in range(4):
            phases[i, 0, i//2, i%2] = 1
        self.register_buffer('phase_kernel', phases)

    def night_phases(self, raw, context):
        m = self.board
        s = m.source
        owner = s.core.model
        body = owner.body(owner.head(s.core.half_input(s.front(raw))))
        ref = owner.global_reference
        c = s.core.half_input(context)
        value = ref.encoder(c)
        if hasattr(ref, 'pyramid'):
            for i in getattr(ref, 'keep', range(len(ref.pyramid))):
                value = value+m.resizes[i](ref.pyramid[i](c).float()).to(value.dtype)
        value = value+value.mean((-2, -1), keepdim=True)
        if self.kind == 'ordinary':
            reference = ref.project(m.sample(value.float()).to(body.dtype))
            return s.output.conv(owner.tail(body+reference))
        reference = m.sample(s.low_ref(ref.project(value)).float()).to(body.dtype)
        return s.output.conv(owner.tail(body))+reference

    def forward(self, raw, context):
        source_raw = raw[:, -self.source_count:]
        phases = (self.night_phases(source_raw, context) if self.kind else
                  self.board.stages(source_raw, context)['preclip_phases'])
        correction = F.conv2d(self.branch(raw), self.phase_kernel, stride=2)
        output = self.board.output if self.kind else self.board.rows
        return output((phases+correction).clamp(0, 1)*255)
