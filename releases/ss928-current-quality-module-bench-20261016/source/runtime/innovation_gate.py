"""Learn a local current-feature blend while retaining the frozen temporal front."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class InnovationConv(nn.Module):
    def __init__(self, old):
        super().__init__()
        assert old.groups == 1 and old.stride[0] == old.stride[1] and old.stride[0] in [1, 2]
        self.multi = copy.deepcopy(old).requires_grad_(False)
        self.current = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding,
                                 dilation=old.dilation, bias=old.bias is not None,
                                 device=old.weight.device, dtype=old.weight.dtype).requires_grad_(False)
        with torch.no_grad():
            self.current.weight.copy_(old.weight.sum(1, keepdim=True))
            if old.bias is not None:
                self.current.bias.copy_(old.bias)
        self.gate = nn.Conv2d(3, 1, 1, device=old.weight.device, dtype=old.weight.dtype)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.register_buffer('mean_kernel', torch.full((1, old.out_channels, 1, 1), 1/old.out_channels,
                                                       device=old.weight.device, dtype=old.weight.dtype))
        self.register_buffer('inverse_scale', torch.ones(1, 3, 1, 1, device=old.weight.device, dtype=old.weight.dtype))
        self.stride_factor = old.stride[0]

    @property
    def weight(self):
        return self.multi.weight

    @property
    def bias(self):
        return self.multi.bias

    def features(self, raw):
        raw = raw.to(self.multi.weight.dtype)
        multi = self.multi(raw)
        current = self.current(raw[:, -1:])
        signal = F.conv2d((current-multi).abs(), self.mean_kernel.to(multi))
        neighbor = F.avg_pool2d(signal, 3, 1, 1, count_include_pad=False)
        recent = raw[:, -1:]
        contrast = (recent-F.avg_pool2d(recent, 3, 1, 1, count_include_pad=False)).abs()
        if self.stride_factor == 2:
            contrast = F.avg_pool2d(contrast, 2, 2)
        values = torch.cat((signal, neighbor, contrast), 1)
        return multi, current, values

    def calibrate(self, raw_values):
        scales = []
        with torch.no_grad():
            for raw in raw_values:
                _, _, features = self.features(raw)
                values = features.permute(1, 0, 2, 3).reshape(3, -1).float()
                scales.append(torch.quantile(values, .90, dim=1).clamp_min(1e-4))
            scale = torch.stack(scales).mean(0).reshape(1, 3, 1, 1)
            self.inverse_scale.copy_(1/scale)

    def forward(self, raw):
        multi, current, features = self.features(raw)
        normalized = (features*self.inverse_scale.to(features)).clamp(0, 128)
        gate = self.gate(normalized).clamp(0, 1)
        return multi+gate*(current-multi)
