"""Provably inactive-channel removal in the preserved nine-frame front."""
import torch
from torch import nn
from torch.nn import functional as F

from reference_candidates import load_candidate as load_reference, prepare_inputs


class TrimmedFront(nn.Module):
    def __init__(self, original, temporal_channels=None, hidden_channels=None):
        super().__init__()
        temporal = original.temporal_conv
        first = original.first
        last = original.last
        t = temporal_channels or temporal.out_channels
        h = hidden_channels or first.out_channels
        assert t in (6, temporal.out_channels)
        assert h <= first.out_channels and h <= last.in_channels
        with torch.no_grad():
            assert torch.count_nonzero(temporal.weight[t:]) == 0
            assert torch.count_nonzero(temporal.bias[t:]) == 0
            assert torch.count_nonzero(first.weight[:, t:]) == 0
            assert torch.count_nonzero(first.weight[h:]) == 0
            assert torch.count_nonzero(first.bias[h:]) == 0
            assert torch.count_nonzero(last.weight[:, h:]) == 0
        device, dtype = first.weight.device, first.weight.dtype
        self.temporal_conv = nn.Conv2d(temporal.in_channels, t, 1,
                                       device=device, dtype=dtype)
        self.first = nn.Conv2d(t, h, first.kernel_size, stride=first.stride,
                               padding=first.padding, device=device, dtype=dtype)
        self.last = nn.Conv2d(h, last.out_channels, last.kernel_size,
                              stride=last.stride, padding=last.padding,
                              device=device, dtype=dtype)
        with torch.no_grad():
            self.temporal_conv.weight.copy_(temporal.weight[:t])
            self.temporal_conv.bias.copy_(temporal.bias[:t])
            self.first.weight.copy_(first.weight[:h, :t])
            self.first.bias.copy_(first.bias[:h])
            self.last.weight.copy_(last.weight[:, :h])
            self.last.bias.copy_(last.bias)

    def forward(self, x):
        x = x.to(dtype=self.first.weight.dtype, memory_format=torch.channels_last)
        return self.last(F.relu(self.first(F.relu(self.temporal_conv(x)))))


def load_front_candidate(scene, name, run_dir, device='cuda'):
    if name in ('rows32', 'keep02_trained', 'keep1_projected'):
        return load_reference(scene, name, run_dir, device=device)
    components = name.split('__')
    front_name = components[0]
    reference_name = components[1] if len(components) > 1 else 'rows32'
    if len(components) > 2 or front_name not in ('trim_t6', 'trim_o20', 'trim_t6_o20'):
        raise ValueError(name)
    model = load_reference(scene, reference_name, run_dir, device=device)
    temporal_channels = 6 if 't6' in front_name else None
    hidden_channels = 20 if 'o20' in front_name else None
    model.front = TrimmedFront(model.front, temporal_channels,
                              hidden_channels).to(device=device).eval()
    return model.eval()
