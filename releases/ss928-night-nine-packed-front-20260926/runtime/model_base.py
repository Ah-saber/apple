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
    def __init__(self, channels, absolute_raw_reference=False, reference_pyramid=False, reference_sensor_y=False):
        super().__init__()
        self.input_channels = 1 + int(absolute_raw_reference) + int(reference_sensor_y)
        self.encoder = nn.Sequential(nn.Conv2d(self.input_channels, 12, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(12, 12, 3, padding=1), nn.GELU())
        if reference_pyramid:
            self.pyramid = nn.ModuleList([nn.Sequential(nn.AvgPool2d(scale),
                nn.Conv2d(self.input_channels,12,3,padding=1),nn.GELU(),
                nn.Conv2d(12,12,3,padding=1)) for scale in (2,4,8)])
            for branch in self.pyramid:
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
        self.project = nn.Conv2d(12, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def encode(self, image):
        features = self.encoder(image)
        if hasattr(self,'pyramid'):
            for branch in self.pyramid:
                features = features + F.interpolate(branch(image),size=(64,64),mode='bilinear',align_corners=False)
        return features

    def forward(self, image, box, size):
        if image is None or box is None:
            raise ValueError('Global-reference model requires RAW thumbnail and crop box')
        if image.ndim != 4 or image.shape[1:] != (self.input_channels, 64, 64) or box.shape != (image.shape[0], 4):
            raise ValueError('Expected matching reference channels at 64x64 and Bx4 box')
        features = self.encode(image)
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
             ('packing_factor', 2), ('output_kernel', 3), ('raw_skip', False), ('absolute_raw_reference', False), ('reference_pyramid',False), ('reference_sensor_y',False), ('input_frames',1), ('temporal_bottleneck',False), ('motion_gate',False), ('trajectory_gate',False)]}


class RT4KSRB0(nn.Module):
    def __init__(self, channels=24, blocks=4, deploy=False, auxiliary_raw=False, global_reference=False,
                 normalization='layernorm', activation='gelu', shuffle_mode='single6', packing_factor=2, output_kernel=3, raw_skip=False, absolute_raw_reference=False, reference_pyramid=False, reference_sensor_y=False, input_frames=1, temporal_bottleneck=False, motion_gate=False, trajectory_gate=False):
        super().__init__()
        if normalization not in ('layernorm', 'none') or activation not in ('gelu', 'relu') or shuffle_mode not in ('single6', 'split23', 'split223') or packing_factor not in (2,4) or output_kernel not in (1,3):
            raise ValueError('Unsupported student architecture option')
        if (packing_factor == 4) != (shuffle_mode == 'split223'):
            raise ValueError('Packing4 requires split223 and packing2 requires single6/split23')
        self.packing_factor, self.output_kernel = packing_factor, output_kernel
        if input_frames not in (1,3,9):raise ValueError("Only one, three or nine input frames supported")
        self.input_frames = input_frames
        self.temporal_bottleneck = bool(temporal_bottleneck)
        self.motion_gate = bool(motion_gate)
        self.trajectory_gate = bool(trajectory_gate)
        if self.trajectory_gate and not self.motion_gate:
            raise ValueError('Trajectory features require motion gate')
        if self.motion_gate and (not self.temporal_bottleneck or input_frames != 9):
            raise ValueError('Motion gate requires nine-frame denoising bottleneck')
        if self.temporal_bottleneck and input_frames not in (3,9):
            raise ValueError('Denoising bottleneck requires three or nine input frames')
        self.raw_skip = bool(raw_skip)
        self.absolute_raw_reference = bool(absolute_raw_reference)
        self.reference_sensor_y = bool(reference_sensor_y)
        if self.reference_sensor_y and not global_reference:raise ValueError("Sensor row requires RAW reference")
        if self.absolute_raw_reference and not global_reference:
            raise ValueError('Absolute RAW reference requires global_reference')
        self.channels, self.blocks, self.deploy = channels, blocks, deploy
        self.normalization, self.activation, self.shuffle_mode = normalization, activation, shuffle_mode
        if self.temporal_bottleneck:
            self.temporal_pre = nn.Sequential(nn.Conv2d(input_frames, 8, 5, padding=2), nn.ReLU(),
                                              nn.Conv2d(8, 1, 5, padding=2))
            nn.init.zeros_(self.temporal_pre[-1].weight)
            nn.init.zeros_(self.temporal_pre[-1].bias)
            if self.motion_gate:
                self.motion_pre = nn.Sequential(nn.Conv2d(input_frames + (4 if self.trajectory_gate else 0), 8, 5, padding=2), nn.ReLU(),
                                                nn.Conv2d(8, 1, 5, padding=2))
                nn.init.zeros_(self.motion_pre[-1].weight)
                nn.init.constant_(self.motion_pre[-1].bias, -3.)
        self.down = nn.PixelUnshuffle(packing_factor)
        head_frames = 1 if self.temporal_bottleneck else input_frames
        self.head = nn.Sequential(nn.Conv2d(head_frames*packing_factor**2, channels, 3, padding=1))
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
            self.global_reference = GlobalReference(channels, absolute_raw_reference, reference_pyramid, reference_sensor_y)

    @staticmethod
    def _shift_zero(image, dy, dx):
        h, w = image.shape[-2:]
        if abs(dy) >= h or abs(dx) >= w:
            return torch.zeros_like(image)
        return F.pad(image[..., max(-dy,0):h-max(dy,0), max(-dx,0):w-max(dx,0)],
                     (max(dx,0),max(-dx,0),max(dy,0),max(-dy,0)))

    def _trajectory_features(self, x):
        """GT-like causal same-polarity support from the latest three corrected frames."""
        reference = x[:, :6].float().median(dim=1, keepdim=True).values
        residual = x.float() - reference
        responses = []
        for kh, kw in ((7,1),(1,7),(3,3)):
            filtered = F.avg_pool2d(F.pad(residual, (kw//2,kw//2,kh//2,kh//2),
                                         mode='replicate'), (kh,kw), stride=1)
            center = filtered.median(dim=1, keepdim=True).values
            scale = (filtered-center).abs().median(dim=1, keepdim=True).values*1.4826
            scale = F.avg_pool2d(F.pad(scale,(3,3,3,3),mode='replicate'),7,stride=1)
            floor = scale.flatten(2).median(dim=2).values[:, :, None, None]*.6
            responses.append(filtered/torch.maximum(scale, floor.clamp_min(.001)))
        z = torch.stack(responses)
        positive = z.amax(dim=0).clamp_min(0).clamp_max(10)
        negative = (-z).amax(dim=0).clamp_min(0).clamp_max(10)
        channels=[]
        velocities=[(0,0)]+[(dy*step,dx*step) for step in (3,6,9,12)
                   for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))]
        for activity in (positive,negative):
            current, previous, older = activity[:, -1:], activity[:, -2:-1], activity[:, -3:-2]
            support = torch.zeros_like(current)
            for dy,dx in velocities:
                pair = self._shift_zero(previous,dy,dx)*self._shift_zero(older,2*dy,2*dx)
                support = torch.maximum(support, torch.sqrt(pair.clamp_min(0)+1e-8))
            channels.extend((current, (current*support).clamp_max(20)))
        return torch.cat(channels,dim=1)

    def forward(self, x, return_auxiliary=False, context=None, context_box=None, input_downsample=1, return_motion_gate=False):
        if input_downsample not in (1,3):raise ValueError("Unsupported RAW sampling scale")
        adapt = input_downsample==3 and hasattr(self,"original_scale_adapter")
        if not torch.jit.is_tracing() and (x.ndim != 4 or x.shape[1] != self.input_frames or
                                           x.shape[-2] % 2 or x.shape[-1] % 2):
            raise ValueError('Input channels must match declared temporal frame count; even spatial dimensions required')
        original = x[:, -1:]
        gate = None
        if self.temporal_bottleneck:
            correction_input = torch.cat((original, *(x[:, j:j+1]-original for j in range(self.input_frames-2, -1, -1))), 1)
            correction = self.temporal_pre(correction_input).float()
            if self.motion_gate:
                gate_input = torch.cat((correction_input, self._trajectory_features(x)),1) if self.trajectory_gate else correction_input
                gate = torch.sigmoid(self.motion_pre(gate_input).float())
            else:
                gate = None
            original = original.float() + correction * (1. - gate if gate is not None else 1.)
            x = original
        if return_motion_gate and (not return_auxiliary or gate is None):
            raise ValueError('Motion gate requires auxiliary output and gated model')
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.packing_factor, (-w) % self.packing_factor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
            if context_box is not None:
                y0, x0, y1, x1 = context_box.unbind(1)
                context_box = torch.stack((y0, x0, y0+(y1-y0)*(h+ph)/h,
                                            x0+(x1-x0)*(w+pw)/w), 1)
        features = self.head(self.down(x))
        if adapt:features=self.original_scale_adapter[0](features)
        features = self.body(features)
        if adapt:features=self.original_scale_adapter[1](features)
        display_features = features
        if hasattr(self, 'global_reference'):
            display_features = features + self.global_reference(context, context_box, features.shape[-2:]).to(features.dtype)
        display_features=self.tail(display_features)
        if adapt:display_features=self.original_scale_adapter[2](display_features)
        display = self.upsample(display_features)[..., :3*h, :3*w]
        if self.raw_skip:
            display = display.float() + F.interpolate(original.float(), size=(3*h,3*w), mode='bilinear', align_corners=False)
        if return_auxiliary:
            if self.temporal_bottleneck:
                return (display, original.float(), gate) if return_motion_gate else (display, original.float())
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
    auxiliary = {k for k in state if k.startswith(('auxiliary_raw.','reference_tone_head.','original_scale_adapter.'))}
    expected = {'auxiliary_raw.0.weight', 'auxiliary_raw.0.bias'} if config.get('auxiliary_raw_weight', 0) and not config.get('temporal_bottleneck', False) else set()
    if config.get('reference_tone_weight',0)>0:
        expected |= {'reference_tone_head.weight','reference_tone_head.bias'}
    if config.get('scale_training_adapter'):
        expected |= {f'original_scale_adapter.{i}.{k}' for i in range(3) for k in ('gain','bias')}
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
    if hasattr(result,'reference_tone_head'):
        del result.reference_tone_head
    if hasattr(result,"original_scale_adapter"):del result.original_scale_adapter
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
