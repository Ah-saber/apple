"""Round 5 candidates. Real quality and timing remain to be measured."""
import torch
from torch import nn
from torch.nn import functional as F


class GroupedSeparableCopyOutput(nn.Module):
    def __init__(self, factor=2, order="width", grouped=True):
        super().__init__()
        if factor not in (2, 4) or order not in ("width", "height"):
            raise ValueError("Unsupported phase factor or order")
        self.factor, self.order, self.grouped = factor, order, grouped
        first = torch.zeros(16, 4, 2, 2)
        for y in range(4):
            for x in range(4):
                first[y*4+x, (y % 2)*2+x % 2, y//2, x//2] = 1
        self.register_buffer("first", first)
        if grouped:
            select = torch.ones(12, 1, 1, 1)
        else:
            select = torch.zeros(12, 4, 1, 1)
            for a in range(2):
                for b in range(6):
                    phase = 2*a+b//3 if order == "width" else 2*(b//3)+a
                    select[6*a+b, phase] = 1
        self.register_buffer("select", select)
        self.register_buffer("repeat", torch.ones(3, 1, 1, 1))

    def forward(self, phases):
        if self.factor == 4:
            phases = F.conv_transpose2d(phases, self.first.to(phases), stride=2)
        n, c, h, w = phases.shape
        if c != 4:
            raise ValueError("Expected four phases")
        if self.grouped and self.order == "height":
            phases = phases.reshape(n, 2, 2, h, w).permute(0, 2, 1, 3, 4).reshape(n, 4, h, w)
        partial = F.conv2d(phases, self.select.to(phases), groups=4 if self.grouped else 1)
        if self.order == "width":
            partial = partial.reshape(n, 2, 6, h, w).permute(0, 3, 1, 4, 2).reshape(n, 1, h*2, w*6)
            final = F.conv2d(partial, self.repeat.to(phases)).permute(0, 2, 1, 3)
        else:
            partial = partial.reshape(n, 2, 6, h, w).permute(0, 3, 2, 4, 1).reshape(n, 1, h*6, w*2)
            final = F.conv2d(partial, self.repeat.to(phases)).permute(0, 2, 3, 1)
        return final.reshape(n, 1, h*6, w*6)


class HistoryGate(nn.Module):
    default_count = 9
    default_detector = "peak"

    def __init__(self, threshold, strength=.75, count=None, force=None):
        super().__init__()
        self.count = self.default_count if count is None else count
        if self.count not in (3, 9) or not 0 <= strength <= 1:
            raise ValueError("Invalid history count or strength")
        self.detector = self.default_detector
        self.strength, self.force = float(strength), force
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def signal(self, x):
        difference = (x[:, -1:] - x[:, -self.count:-1]).abs()
        if self.detector == "peak":
            return F.max_pool2d(difference, 3, 1, 1)
        return F.avg_pool2d(difference, 3, 1, 1, count_include_pad=False)

    def forward(self, x):
        if self.force == 0:
            return x
        tau = self.threshold.to(x).clamp_min(1e-6)
        gate = ((self.signal(x) - tau) / tau).clamp(0, 1) * self.strength
        past = x[:, -self.count:-1]
        adjusted = past + gate * (x[:, -1:] - past)
        return torch.cat((x[:, :-self.count], adjusted, x[:, -1:]), 1)
