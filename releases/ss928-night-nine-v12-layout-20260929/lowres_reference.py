"""Move the reference contribution into four low-resolution phase channels."""
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from front_candidates import load_front_candidate, prepare_inputs
from output_candidates import OutputLayout


def tail_conv(tail):
    if isinstance(tail, nn.Sequential) and len(tail) == 2 and isinstance(tail[0], nn.Identity):
        tail = tail[1].rep_conv
    assert isinstance(tail, nn.Conv2d) and tail.kernel_size == (3, 3)
    return tail


def compose_center(tail, projection):
    tail = tail_conv(tail)
    assert isinstance(tail, nn.Conv2d) and tail.kernel_size == (3, 3)
    assert isinstance(projection, nn.Conv2d) and projection.kernel_size == (3, 3)
    return projection.weight[:4].float().sum((-1, -2)) @ tail.weight.float().sum((-1, -2))


class LowResReference(nn.Module):
    def __init__(self, base, layout='rows32', kernel=3):
        super().__init__()
        assert kernel in (1, 3, 5)
        self.front = base.front
        self.core = base.core
        self.output = OutputLayout(base.output, layout)
        self.kernel = kernel
        owner = self.core.model
        self.low_ref = nn.Conv2d(16, 4, kernel, padding=kernel // 2,
            bias=False, device=self.output.conv.weight.device,
            dtype=self.output.conv.weight.dtype)
        with torch.no_grad():
            self.low_ref.weight.zero_()
            self.low_ref.weight[:, :, kernel // 2, kernel // 2].copy_(
                compose_center(owner.tail, self.output.conv).to(
                    dtype=self.low_ref.weight.dtype))

    def reference_features(self, context):
        owner = self.core.model
        reference = owner.global_reference.encode(self.core.half_input(context))
        reference = reference + reference.mean((-2, -1), keepdim=True)
        return owner.global_reference.project(reference)

    def forward(self, raw, context):
        owner = self.core.model
        body = owner.body(owner.head(self.core.half_input(self.front(raw))))
        reference = self.reference_features(context)
        contribution = self.low_ref(reference)
        contribution = F.interpolate(contribution, size=body.shape[-2:],
                                     mode='bilinear', align_corners=False)
        body_phases = self.output.conv(owner.tail(body))
        phases = (body_phases + contribution).clamp(0, 1) * 255
        return self.output.finish(phases)


def load_candidate(scene, kernel, layout, run_dir, device='cuda', trained=False):
    stem = ('trim_t6__keep02_trained' if scene == 'ordinary'
            else 'trim_t6_o20')
    base = load_front_candidate(scene, stem, run_dir, device=device)
    model = LowResReference(base, layout=layout, kernel=kernel).to(device=device).eval()
    if trained:
        checkpoint_root = Path(os.environ.get('RAWIR_V12_CHECKPOINTS',
            str(Path(run_dir).parent / 'SS928-BOARD-V12-20260929')))
        state = torch.load(checkpoint_root / f'{scene}_lowref_k{kernel}.pt',
                           map_location='cpu', weights_only=True)
        model.low_ref.load_state_dict(state['low_ref'])
    return model.eval()


def teacher_reference_phases(model, context, feature_shape=(512, 640)):
    owner = model.core.model
    reference = model.reference_features(context)
    reference = F.interpolate(reference.float(), size=feature_shape,
                              mode='bilinear', align_corners=False)
    tail = tail_conv(owner.tail)
    projection = model.output.conv
    value = F.conv2d(reference, tail.weight.float(), bias=None, padding=1)
    return F.conv2d(value, projection.weight.float(), bias=None, padding=1)
