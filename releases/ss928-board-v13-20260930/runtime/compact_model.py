"""Quarter-grid, sixteen-phase model with a separate low-frequency reference."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class CompactModel(nn.Module):
    factor = 4

    def __init__(self, reference, width=16, depth=1, raw_skip=False, late_reference=True):
        super().__init__()
        self.width, self.depth, self.raw_skip = width, depth, raw_skip
        self.late_reference = late_reference
        self.front = nn.Conv2d(9, width, 6, stride=4, padding=1)
        layers = []
        for _ in range(depth):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU()])
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 16, 3, padding=1)
        self.reference = copy.deepcopy(reference)
        old = self.reference.project
        self.reference.project = nn.Conv2d(old.in_channels, 16 if late_reference else width, 1)
        nn.init.zeros_(self.reference.project.weight)
        nn.init.zeros_(self.reference.project.bias)
        if raw_skip:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        # Sampling the last RAW frame is a fixed convolution, avoiding
        # SpaceToDepth on the full-resolution residual branch.
        kernel = torch.zeros(16, 9, 4, 4)
        for i in range(16):
            kernel[i, 8, i // 4, i % 4] = 1
        self.register_buffer('raw_kernel', kernel)

    def phases(self, stack, context, box):
        dtype = self.front.weight.dtype
        stack = stack.to(dtype=dtype, memory_format=torch.channels_last)
        x = F.relu(self.front(stack))
        reference = self.reference(context.to(dtype=dtype), box, x.shape[-2:]).to(dtype=dtype)
        x = self.head(self.body(x)) + reference if self.late_reference else self.head(self.body(x+reference))
        if self.raw_skip:
            x = x + F.conv2d(stack, self.raw_kernel.to(dtype), stride=4)
        return x

    def native(self, stack, context, box):
        return F.pixel_shuffle(self.phases(stack, context, box), 4)

    def forward(self, stack, context, box):
        return F.interpolate(self.native(stack, context, box).clamp(0, 1)*255,
                             scale_factor=3, mode='nearest')


class FixedSample(nn.Module):
    """Frozen bilinear/border/align_corners=False crop, without GridSample."""
    def __init__(self, box, height, width, source=64):
        super().__init__()
        box = box.detach().flatten().float()
        for axis, size, lo, hi in [('y', height, box[0], box[2]),
                                  ('x', width, box[1], box[3])]:
            unit = (torch.arange(size,device=box.device).float()+.5)/size*(hi-lo)+lo
            # Preserve GridSample's normalized-coordinate round trip, including
            # rounding for C32 boxes that extend far outside the unit interval.
            grid = unit*2-1
            coord = ((grid+1)*source-1)*.5
            coord = coord.clamp(0, source-1)
            index = coord.floor().long()
            self.register_buffer(axis+'0', index)
            self.register_buffer(axis+'1', (index+1).clamp_max(source-1))
            shape = (1, 1, size, 1) if axis == 'y' else (1, 1, 1, size)
            self.register_buffer(axis+'w', (coord-index).reshape(shape))

    def forward(self, x):
        y = x.index_select(2, self.y0)*(1-self.yw) + x.index_select(2, self.y1)*self.yw
        return y.index_select(3, self.x0)*(1-self.xw) + y.index_select(3, self.x1)*self.xw


class Rows32(nn.Module):
    """Verified 32-row output ordering, preserving all sixteen phases."""
    def __init__(self):
        super().__init__()
        # First reconstruct the half-grid four phases with a fixed stride-2
        # convolution transpose. The existing rows32 kernel expands 6x.
        first = torch.zeros(16, 4, 2, 2)
        for py in range(4):
            for px in range(4):
                first[py*4+px, (py % 2)*2+px % 2, py//2, px//2] = 1
        kernel = torch.zeros(64, 32, 3, 6)
        for fy in range(16):
            for py in range(2):
                for px in range(2):
                    for sy in range(3):
                        yy = fy*6+py*3+sy
                        for sx in range(3):
                            kernel[fy*4+py*2+px, yy % 32, yy//32, px*3+sx] = 1
        self.register_buffer('first', first)
        self.register_buffer('kernel', kernel)

    def forward(self, phases):
        half = F.conv_transpose2d(phases, self.first.to(phases.dtype), stride=2)
        n, c, h, w = half.shape
        packed = half.reshape(n, c, h//16, 16, w).permute(0, 3, 1, 2, 4)
        packed = packed.reshape(n, c*16, h//16, w)
        blocked = F.conv_transpose2d(packed, self.kernel.to(phases.dtype), stride=(3, 6))
        return blocked.permute(0, 2, 1, 3).reshape(n, 1, h*6, w*6)


class BoardCompact(nn.Module):
    def __init__(self, model, box, layout='rows32', height=1024, width=1280):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sample = FixedSample(box, height//4, width//4)
        self.resizes = nn.ModuleList([FixedSample(torch.tensor([[0.,0.,1.,1.]]),64,64,64//s)
                                     for s in (2,4,8)])
        self.output = Rows32()
        self.layout = layout

    def forward(self, stack, context):
        m = self.model
        stack = stack.to(m.front.weight.dtype)
        x = F.relu(m.front(stack))
        context = context.to(m.front.weight.dtype)
        ref = m.reference.encoder(context)
        if hasattr(m.reference,'pyramid'):
            for i,branch in enumerate(m.reference.pyramid):
                ref = ref + self.resizes[i](branch(context).float()).to(ref.dtype)
        ref = m.reference.project(ref + ref.mean((-2, -1), keepdim=True))
        ref = self.sample(ref.float()).to(x.dtype)
        x = m.head(m.body(x)) + ref if m.late_reference else m.head(m.body(x+ref))
        if m.raw_skip:
            x = x + F.conv2d(stack, m.raw_kernel.to(x.dtype), stride=4)
        phases = x.clamp(0, 1)*255
        return phases if self.layout == 'phases' else self.output(phases)
