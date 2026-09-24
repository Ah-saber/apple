"""RT4KSR B0 adaptation, derived from Zamfir et al. RT4KSR (Apache-2.0).

Source commit fd6627a48d789adf5d9aad29f02ab2a3d2a25296. Modified for
one-channel RAW/display x3, output-reachable factory graph only. No pretrained
weights. See third_party/RT4KSR/LICENSE and PROVENANCE.json.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Same channel population statistics as upstream; keep AMP statistics FP32.
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        y = (value - mean) * torch.rsqrt(variance + self.eps)
        y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return y.to(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, channels, ratio=2):
        super().__init__()
        expanded = channels * ratio
        self.expand_conv = nn.Conv2d(channels, expanded, 1)
        self.fea_conv = nn.Conv2d(expanded, expanded, 3)
        self.reduce_conv = nn.Conv2d(expanded, channels, 1)

    def forward(self, x):
        expanded = self.expand_conv(x)
        padded = F.pad(expanded, (1, 1, 1, 1))
        bias = self.expand_conv.bias[None, :, None, None]
        padded[:, :, :1, :] = bias
        padded[:, :, -1:, :] = bias
        padded[:, :, :, :1] = bias
        padded[:, :, :, -1:] = bias
        return self.reduce_conv(self.fea_conv(padded) + expanded) + x


class RepResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rep_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.rep_conv(x)


class Block(nn.Module):
    def __init__(self, channels, deploy=False, normalization='layernorm', activation='gelu'):
        super().__init__()
        self.norm = LayerNorm2d(channels) if normalization == 'layernorm' else nn.Identity()
        self.conv1 = RepResBlock(channels) if deploy else ResBlock(channels)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv1(self.norm(x)))


class GlobalReference(nn.Module):
    """RAW thumbnail features aligned to the crop; zero-initialized display correction."""
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(12, 12, 3, padding=1), nn.GELU())
        self.project = nn.Conv2d(12, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, image, box, size):
        if image is None or box is None:
            raise ValueError('Global-reference model requires RAW thumbnail and crop box')
        if image.ndim != 4 or image.shape[1:] != (1, 64, 64) or box.shape != (image.shape[0], 4):
            raise ValueError('Expected Bx1x64x64 thumbnail and Bx4 box')
        features = self.encoder(image)
        features = self.project(features + features.mean((-2, -1), keepdim=True))
        h, w = size
        yy = (torch.arange(h, device=box.device, dtype=torch.float32) + .5) / h
        xx = (torch.arange(w, device=box.device, dtype=torch.float32) + .5) / w
        y0, x0, y1, x1 = box.float().unbind(1)
        gy = y0[:, None, None] + yy[None, :, None] * (y1-y0)[:, None, None]
        gx = x0[:, None, None] + xx[None, None, :] * (x1-x0)[:, None, None]
        grid = torch.stack((gx.expand(-1,h,-1)*2-1, gy.expand(-1,-1,w)*2-1), -1)
        return F.grid_sample(features.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)


def staged_shuffle_permutation(factors):
    # Reverse the channel/phase decomposition induced by successive PixelShuffles.
    scale = math.prod(factors)
    result = []
    for channel in range(scale * scale):
        q, row, col = channel, 0, 0
        for factor in factors:
            phase = q % (factor * factor)
            q //= factor * factor
            row, col = row * factor + phase // factor, col * factor + phase % factor
        result.append(row * scale + col)
    return result


@torch.no_grad()
def use_staged_shuffle(model, factors):
    scale = math.prod(factors)
    if len(model.upsample) != 2 or model.upsample[1].upscale_factor != scale:
        raise ValueError('Output scale mismatch')
    conv = model.upsample[0]
    perm = staged_shuffle_permutation(factors)
    conv.weight.copy_(conv.weight[perm].clone())
    conv.bias.copy_(conv.bias[perm].clone())
    model.upsample = nn.Sequential(conv, *[nn.PixelShuffle(f) for f in factors])
    return model


def shuffle23_permutation():
    # CRD phase order from the verified SS928 a01 graph rewrite.
    return [(3*r1+r2)*6+3*c1+c2
            for r2 in range(3) for c2 in range(3)
            for r1 in range(2) for c1 in range(2)]


@torch.no_grad()
def use_shuffle23(model):
    if len(model.upsample) != 2 or model.upsample[1].upscale_factor != 6:
        raise ValueError('Expected the original single Shuffle6 output')
    conv = model.upsample[0]
    perm = shuffle23_permutation()
    conv.weight.copy_(conv.weight[perm].clone())
    conv.bias.copy_(conv.bias[perm].clone())
    model.upsample = nn.Sequential(conv, nn.PixelShuffle(2), nn.PixelShuffle(3))
    model.shuffle_mode = 'split23'
    return model


def architecture_options(config):
    return {k: config.get(k, default) for k, default in
            [('normalization', 'layernorm'), ('activation', 'gelu'), ('shuffle_mode', 'single6'),
             ('packing_factor', 2), ('output_kernel', 3), ('raw_skip', False)]}


class RT4KSRB0(nn.Module):
    def __init__(self, channels=24, blocks=4, deploy=False, auxiliary_raw=False, global_reference=False,
                 normalization='layernorm', activation='gelu', shuffle_mode='single6', packing_factor=2, output_kernel=3, raw_skip=False):
        super().__init__()
        if normalization not in ('layernorm', 'none') or activation not in ('gelu', 'relu') or shuffle_mode not in ('single6', 'split23', 'split223') or packing_factor not in (2,4) or output_kernel not in (1,3):
            raise ValueError('Unsupported student architecture option')
        if (packing_factor == 4) != (shuffle_mode == 'split223'):
            raise ValueError('Packing4 requires split223 and packing2 requires single6/split23')
        self.packing_factor, self.output_kernel = packing_factor, output_kernel
        self.raw_skip = bool(raw_skip)
        self.channels, self.blocks, self.deploy = channels, blocks, deploy
        self.normalization, self.activation, self.shuffle_mode = normalization, activation, shuffle_mode
        self.down = nn.PixelUnshuffle(packing_factor)
        self.head = nn.Sequential(nn.Conv2d(packing_factor**2, channels, 3, padding=1))
        self.body = nn.Sequential(*[Block(channels, deploy, normalization, activation) for _ in range(blocks)])
        self.tail = nn.Sequential(LayerNorm2d(channels) if normalization == 'layernorm' else nn.Identity(),
                                  RepResBlock(channels) if deploy else ResBlock(channels))
        self.upsample = nn.Sequential(nn.Conv2d(channels, (3*packing_factor)**2, output_kernel, padding=output_kernel//2), nn.PixelShuffle(3*packing_factor))
        if raw_skip:
            nn.init.zeros_(self.upsample[0].weight)
            nn.init.zeros_(self.upsample[0].bias)
        if shuffle_mode == 'split23':
            use_shuffle23(self)
        elif shuffle_mode == 'split223':
            use_staged_shuffle(self, (2,2,3))
        if auxiliary_raw:
            self.auxiliary_raw = nn.Sequential(nn.Conv2d(channels, packing_factor**2, 3, padding=1), nn.PixelShuffle(packing_factor))
            nn.init.zeros_(self.auxiliary_raw[0].weight)
            nn.init.zeros_(self.auxiliary_raw[0].bias)
        if global_reference:
            self.global_reference = GlobalReference(channels)

    def forward(self, x, return_auxiliary=False, context=None, context_box=None):
        if not torch.jit.is_tracing() and (x.ndim != 4 or x.shape[1] != 1 or
                                           x.shape[-2] % 2 or x.shape[-1] % 2):
            raise ValueError('B0 requires NCHW single-channel input with even spatial dimensions')
        original = x
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.packing_factor, (-w) % self.packing_factor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
            if context_box is not None:
                y0, x0, y1, x1 = context_box.unbind(1)
                context_box = torch.stack((y0, x0, y0+(y1-y0)*(h+ph)/h,
                                            x0+(x1-x0)*(w+pw)/w), 1)
        features = self.body(self.head(self.down(x)))
        display_features = features
        if hasattr(self, 'global_reference'):
            display_features = features + self.global_reference(context, context_box, features.shape[-2:]).to(features.dtype)
        display = self.upsample(self.tail(display_features))[..., :3*h, :3*w]
        if self.raw_skip:
            display = display.float() + F.interpolate(original.float(), size=(3*h,3*w), mode='bilinear', align_corners=False)
        if return_auxiliary:
            if not hasattr(self, 'auxiliary_raw'):
                raise ValueError('No auxiliary RAW head configured')
            # Keep the small DN correction in FP32, including under BF16 autocast.
            middle = original.float() + self.auxiliary_raw(features)[..., :h, :w].float()
            return display, middle
        return display


def inference_model(config, state):
    """Strictly load the original inference graph, excluding only the declared training head."""
    model = RT4KSRB0(config['channels'], config['blocks'], global_reference=config.get('global_reference', False),
                    **architecture_options(config))
    auxiliary = {k for k in state if k.startswith('auxiliary_raw.')}
    expected = {'auxiliary_raw.0.weight', 'auxiliary_raw.0.bias'} if config.get('auxiliary_raw_weight', 0) else set()
    if auxiliary != expected:
        raise ValueError('Auxiliary checkpoint/config identity mismatch')
    model.load_state_dict({k: v for k, v in state.items() if k not in auxiliary}, strict=True)
    return model


@torch.no_grad()
def fuse_block(source):
    expand = source.expand_conv.weight[:, :, 0, 0]
    expand_bias = source.expand_conv.bias
    middle = source.fea_conv.weight.clone()
    index = torch.arange(middle.shape[0], device=middle.device)
    middle[index, index, 1, 1] += 1
    reduce = source.reduce_conv.weight[:, :, 0, 0]
    weight = torch.einsum('oa,abij,bc->ocij', reduce, middle, expand)
    index = torch.arange(weight.shape[0], device=weight.device)
    weight[index, index, 1, 1] += 1
    bias = reduce @ (middle.sum((2, 3)) @ expand_bias + source.fea_conv.bias) + source.reduce_conv.bias
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1).to(weight)
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)
    return conv


@torch.no_grad()
def to_deploy(model):
    result = copy.deepcopy(model).eval()
    if hasattr(result, 'auxiliary_raw'):
        del result.auxiliary_raw
    if model.deploy:
        return result
    for block in result.body:
        replacement = RepResBlock(result.channels).to(next(block.parameters()))
        replacement.rep_conv = fuse_block(block.conv1)
        block.conv1 = replacement
    replacement = RepResBlock(result.channels).to(next(result.tail.parameters()))
    replacement.rep_conv = fuse_block(result.tail[1])
    result.tail[1] = replacement
    result.deploy = True
    return result
