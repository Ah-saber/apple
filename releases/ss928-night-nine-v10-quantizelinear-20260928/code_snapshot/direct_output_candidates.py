"""Exact FP16 complete-output controls for the v0.9 board hotspot.

The old graphs and weights remain untouched. These candidates replace the
blocked-output restore (Concat/Transpose/Reshape) with one direct scatter
convolution. Board timing and operator placement must be measured separately.
"""
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

V09 = Path(os.environ.get('RAWIR_V09_CODE',
    str(Path(__file__).resolve().parents[1] / 'ss928_boarddriven_v09_20260928')))
sys.path.insert(0, str(V09))
from board_candidates import load_board, prepare_inputs  # noqa: E402


class DirectRowsOutput(nn.Module):
    """Place each two-by-two phase directly in its sixfold display block."""

    def __init__(self, old, group):
        super().__init__()
        if group not in (1, 2, 4, 8, 16):
            raise ValueError(group)
        self.conv = old.conv
        self.group = group
        weight = torch.zeros(
            (4 * group, 1, 6 * group, 6),
            device=self.conv.weight.device,
            dtype=self.conv.weight.dtype,
        )
        for row in range(group):
            for phase_y in range(2):
                for phase_x in range(2):
                    channel = row * 4 + phase_y * 2 + phase_x
                    weight[channel, 0, row * 6 + phase_y * 3:row * 6 + phase_y * 3 + 3,
                           phase_x * 3:phase_x * 3 + 3] = 1
        self.register_buffer('weight', weight)

    def forward(self, features):
        phases = self.conv(features.to(dtype=self.conv.weight.dtype,
                                       memory_format=torch.channels_last))[:, :4].clamp(0, 1) * 255
        n, c, h, w = (int(v) for v in phases.shape)
        if h % self.group:
            raise ValueError('height must be divisible by group')
        if self.group == 1:
            packed = phases
        else:
            packed = phases.reshape(n, c, h // self.group, self.group, w)
            packed = packed.permute(0, 3, 1, 2, 4).reshape(n, c * self.group, h // self.group, w)
        return F.conv_transpose2d(packed, self.weight, stride=(6 * self.group, 6))


def load_candidate(scene, name, run_dir, device='cuda'):
    if name == 'rows32':
        return load_board(scene, name, run_dir, device=device)
    if not name.startswith('direct_g'):
        raise ValueError(name)
    group = int(name.removeprefix('direct_g'))
    model = load_board(scene, 'rows32', run_dir, device=device)
    model.output = DirectRowsOutput(model.output, group).to(device=device).eval()
    return model.eval()
