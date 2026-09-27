"""Exact output permutations and decomposed nine-frame statistics, no new weights."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class OutputPermutation(nn.Module):
    def __init__(self, mode='shuffle6'):
        super().__init__()
        self.mode = mode
        factor = 6 if mode == 'deconv6' else (2 if mode == 'deconv2_shuffle3' else 3)
        if mode.startswith('deconv'):
            rest = 6 // factor
            weight = torch.zeros(36, rest * rest, factor, factor)
            for dy in range(factor):
                for dx in range(factor):
                    for sy in range(rest):
                        for sx in range(rest):
                            weight[(rest * dy + sy) * 6 + rest * dx + sx, sy * rest + sx, dy, dx] = 1
            self.register_buffer('weight', weight)
            self.factor = factor
        if mode in ('shuffle2_3', 'shuffle3_2'):
            first = 2 if mode == 'shuffle2_3' else 3
            rest = 6 // first
            ids = [(rest * dy + sy) * 6 + rest * dx + sx
                   for sy in range(rest) for sx in range(rest)
                   for dy in range(first) for dx in range(first)]
            self.register_buffer('ids', torch.tensor(ids, dtype=torch.long))
            self.factor = first

    def forward(self, packed):
        if self.mode == 'shuffle6':
            return F.pixel_shuffle(packed, 6)
        if self.mode == 'reshape':
            b, channels, h, w = (int(v) for v in packed.shape)
            if b != 1 or channels != 36:
                raise ValueError('Expected static batch-one 36-phase input')
            return packed.reshape(6, 6, h, w).permute(2, 0, 3, 1).reshape(1, 1, 6*h, 6*w)
        if self.mode.startswith('deconv'):
            out = F.conv_transpose2d(packed, self.weight.to(dtype=packed.dtype), stride=self.factor)
            return out if self.factor == 6 else F.pixel_shuffle(out, 6//self.factor)
        if self.mode.startswith('shuffle'):
            out = F.pixel_shuffle(packed[:, self.ids], self.factor)
            return F.pixel_shuffle(out, 6//self.factor)
        raise ValueError(self.mode)

class Statistics(nn.Module):
    def __init__(self, old_weight, mode='dense'):
        super().__init__()
        self.mode = mode
        self.register_buffer('original', old_weight.detach().clone())
        temporal = torch.stack([old_weight[4*i, :, 0, 0] for i in range(3)])[:, :, None, None]
        # Verify every phase uses the same temporal coefficients and one spatial tap.
        expected = torch.zeros_like(old_weight)
        for group in range(3):
            for dy in range(2):
                for dx in range(2):
                    expected[group*4+dy*2+dx, :, dy, dx] = temporal[group, :, 0, 0]
        if not torch.equal(expected, old_weight):
            raise ValueError('Unrecognized statistical kernel')
        self.register_buffer('temporal', temporal)
        pack = torch.zeros(12, 1, 2, 2, dtype=old_weight.dtype, device=old_weight.device)
        pack9 = torch.zeros(36, 1, 2, 2, dtype=old_weight.dtype, device=old_weight.device)
        packed_mix = torch.zeros(12, 36, 1, 1, dtype=old_weight.dtype, device=old_weight.device)
        for dy in range(2):
            for dx in range(2):
                phase = dy*2+dx
                for t in range(9):
                    pack9[t*4+phase, 0, dy, dx] = 1
                for group in range(3):
                    pack[group*4+phase, 0, dy, dx] = 1
                    packed_mix[group*4+phase, phase::4, 0, 0] = temporal[group, :, 0, 0]
        self.register_buffer('pack', pack)
        self.register_buffer('pack9', pack9)
        self.register_buffer('packed_mix', packed_mix)

    def forward(self, stack):
        x = stack.to(dtype=self.original.dtype, memory_format=torch.channels_last)
        if self.mode == 'dense':
            return F.conv2d(x, self.original, stride=2)
        if self.mode == 'temporal_first':
            return F.conv2d(F.conv2d(x, self.temporal), self.pack, stride=2, groups=3)
        if self.mode == 'temporal_f32':
            # Round only after both operations; separate from half intermediate.
            return F.conv2d(F.conv2d(x.float(), self.temporal.float()), self.pack.float(), stride=2, groups=3).to(self.original.dtype)
        if self.mode == 'pack_first':
            return F.conv2d(F.conv2d(x, self.pack9, stride=2, groups=9), self.packed_mix)
        raise ValueError(self.mode)

class RewrittenFront(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.stats = Statistics(original.stats_weight, mode)
        self.first = copy.deepcopy(original.first)
        self.last = copy.deepcopy(original.last)
    def forward(self, stack):
        stats = self.stats(stack)
        current, old, recent = stats[:, :4].float(), stats[:, 4:8].float(), stats[:, 8:].float()
        features = torch.cat(((current-old)*64, (current-recent)*64, current), 1).to(dtype=self.first.weight.dtype, memory_format=torch.channels_last)
        return current + self.last(F.relu(self.first(features))).float()*.025

class CandidateSystem(nn.Module):
    def __init__(self, original, statistics='dense', output='shuffle6', input_half=False, output_half=False, scale='full_f32'):
        super().__init__()
        self.core = original.core
        self.front = original.front if statistics == 'dense' else RewrittenFront(original.front, statistics)
        self.permutation = OutputPermutation(output).to(device=original.front.first.weight.device, dtype=torch.float16)
        self.input_half, self.output_half, self.scale = input_half, output_half, scale
    def forward(self, stack, context):
        c, m = self.core, self.core.model
        features = m.body(m.head(c.half_input(self.front(stack))))
        ref = m.global_reference.encode(c.half_input(context))
        ref = m.global_reference.project(ref+ref.mean((-2,-1), keepdim=True))
        ref = F.interpolate(ref, size=features.shape[-2:], mode='bilinear', align_corners=False)
        packed = m.upsample[0](m.tail(features+ref))
        if self.scale == 'packed_f32':
            gray = self.permutation(packed.float().clamp(0,1)*255)
        elif self.scale == 'packed_half':
            gray = self.permutation(packed.clamp(0,1)*255).float()
        else:
            gray = self.permutation(packed).float().clamp(0,1)*255
        return gray.half() if self.output_half else gray

class FoldedStatisticsFront(nn.Module):
    """Compose statistics/differences/first learning Conv into spatial Conv.
    Floating algebra is equivalent; half rounding order changes and is measured.
    Current four-phase skip is extracted from current frame alone.
    """
    def __init__(self, original, precision='float16'):
        super().__init__()
        first = original.first
        old = first.weight.detach().float()
        temporal = original.stats_weight.detach().float()
        weight = torch.zeros((old.shape[0],9,6,6), device=old.device)
        for dy in range(2):
            for dx in range(2):
                phase=2*dy+dx
                for ky in range(3):
                    for kx in range(3):
                        current=old[:,phase,ky,kx]*64+old[:,4+phase,ky,kx]*64+old[:,8+phase,ky,kx]
                        weight[:,8,2*ky+dy,2*kx+dx]+=current
                        for t in range(9):
                            weight[:,t,2*ky+dy,2*kx+dx]-=old[:,phase,ky,kx]*64*temporal[4+phase,t,dy,dx]+old[:,4+phase,ky,kx]*64*temporal[8+phase,t,dy,dx]
        dtype=torch.float32 if precision=='float32' else torch.float16
        self.register_buffer('weight',weight.to(dtype))
        self.register_buffer('bias',first.bias.detach().to(dtype))
        self.last=copy.deepcopy(original.last)
        pack=torch.zeros(4,1,2,2,device=old.device,dtype=torch.float16)
        for dy in range(2):
            for dx in range(2):pack[2*dy+dx,0,dy,dx]=1
        self.register_buffer('pack',pack)
    def forward(self, stack):
        x=stack.to(dtype=torch.float16,memory_format=torch.channels_last)
        current=F.conv2d(x[:,-1:],self.pack,stride=2).float()
        features=F.relu(F.conv2d(x.to(self.weight.dtype),self.weight,self.bias,stride=2,padding=2)).to(self.last.weight.dtype)
        return current+self.last(features).float()*.025
