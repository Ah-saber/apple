"""Full-frame output layouts for SS928, preserving four learned pixel phases."""
import torch
from torch import nn
from torch.nn import functional as F

from output_layout_candidates import load_output_candidate, prepare_inputs
from front_candidates import load_front_candidate


def four_channel_conv(source):
    assert isinstance(source, nn.Conv2d) and source.out_channels >= 4
    conv = nn.Conv2d(source.in_channels, 4, source.kernel_size,
                     stride=source.stride, padding=source.padding,
                     dilation=source.dilation, groups=source.groups,
                     bias=source.bias is not None,
                     device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        conv.weight.copy_(source.weight[:4])
        if source.bias is not None:
            conv.bias.copy_(source.bias[:4])
    return conv


class OutputLayout(nn.Module):
    def __init__(self, old, layout):
        super().__init__()
        self.conv = four_channel_conv(old.conv)
        self.layout = layout
        if layout == 'native_deconv':
            self.register_buffer('kernel', torch.ones(1, 1, 3, 3,
                device=self.conv.weight.device, dtype=self.conv.weight.dtype))
        elif layout == 'native_separable':
            self.register_buffer('kernel_h', torch.ones(1, 1, 3, 1,
                device=self.conv.weight.device, dtype=self.conv.weight.dtype))
            self.register_buffer('kernel_w', torch.ones(1, 1, 1, 3,
                device=self.conv.weight.device, dtype=self.conv.weight.dtype))
        elif layout == 'direct6':
            kernel = torch.zeros(4, 1, 6, 6,
                device=self.conv.weight.device, dtype=self.conv.weight.dtype)
            for py in range(2):
                for px in range(2):
                    kernel[py * 2 + px, 0, py * 3:py * 3 + 3,
                           px * 3:px * 3 + 3] = 1
            self.register_buffer('kernel', kernel)
        elif layout.startswith('rows'):
            rows = int(layout[4:])
            assert rows in (8, 16, 32, 64, 128)
            group = rows // 2
            kernel = torch.zeros(group * 4, rows, 3, 6,
                device=self.conv.weight.device, dtype=self.conv.weight.dtype)
            for fy in range(group):
                for py in range(2):
                    for px in range(2):
                        channel = fy * 4 + py * 2 + px
                        for sy in range(3):
                            yy = fy * 6 + py * 3 + sy
                            for sx in range(3):
                                kernel[channel, yy % rows, yy // rows,
                                       px * 3 + sx] = 1
            self.register_buffer('kernel', kernel)
            self.rows = rows
            self.group = group
        elif layout != 'native_nearest':
            raise ValueError(layout)

    def forward(self, value):
        phases = self.conv(value.to(dtype=self.conv.weight.dtype,
            memory_format=torch.channels_last)).clamp(0, 1) * 255
        return self.finish(phases)

    def finish(self, phases):
        if self.layout == 'direct6':
            return F.conv_transpose2d(phases, self.kernel, stride=6)
        if self.layout.startswith('rows'):
            n, c, h, w = (int(v) for v in phases.shape)
            assert h % self.group == 0
            packed = phases.reshape(n, c, h // self.group, self.group, w)
            packed = packed.permute(0, 3, 1, 2, 4).reshape(
                n, c * self.group, h // self.group, w)
            blocked = F.conv_transpose2d(packed, self.kernel, stride=(3, 6))
            return blocked.permute(0, 2, 1, 3).reshape(n, 1, h * 6, w * 6)
        native = F.pixel_shuffle(phases, 2)
        if self.layout == 'native_deconv':
            return F.conv_transpose2d(native, self.kernel, stride=3)
        if self.layout == 'native_separable':
            value = F.conv_transpose2d(native, self.kernel_h, stride=(3, 1))
            return F.conv_transpose2d(value, self.kernel_w, stride=(1, 3))
        return F.interpolate(native, scale_factor=3, mode='nearest')


def load_candidate(scene, layout, run_dir, device='cuda'):
    stem = ('trim_t6__keep02_trained' if scene == 'ordinary'
            else 'trim_t6_o20')
    model = load_front_candidate(scene, stem, run_dir, device=device)
    model.output = OutputLayout(model.output, layout).to(device=device).eval()
    return model.eval()


def load_v11_expanded(scene, run_dir, device='cuda'):
    name = ('trim_t6__keep02_trained__expand4' if scene == 'ordinary'
            else 'trim_t6_o20__expand4')
    return load_output_candidate(scene, name, run_dir, device=device)
