"""Isolated reference-stage ablations for full nine-frame SS928 models.

Only output-equivalent variants may be delivered without retraining. Ablations
measure each pyramid branch's quality contribution before any distillation.
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


class SelectPyramid(nn.Module):
    def __init__(self, original, keep):
        super().__init__()
        self.encoder = original.encoder
        self.pyramid = nn.ModuleList(original.pyramid)
        self.project = original.project
        self.keep = tuple(keep)

    def encode(self, x):
        value = self.encoder(x)
        for index in self.keep:
            value = value + F.interpolate(self.pyramid[index](x), size=(64, 64),
                                          mode='bilinear', align_corners=False)
        return value


class ShallowEncoderReference(nn.Module):
    def __init__(self, original, kernel):
        super().__init__()
        self.project = original.project
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel, padding=kernel // 2,
                      device=self.project.weight.device, dtype=self.project.weight.dtype),
            nn.ReLU(),
        )

    def encode(self, x):
        return self.encoder(x)


class ResizeReferenceSystem(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original = original
        self.mode = mode
        self.input_channels = original.input_channels if hasattr(original, 'input_channels') else 9
        self.input_half = getattr(original, 'input_half', True)

    def forward(self, x, context):
        old = self.original
        core = old.core
        owner = core.model
        value = owner.body(owner.head(core.half_input(old.front(x))))
        reference = owner.global_reference.encode(core.half_input(context))
        reference = reference + reference.mean((-2, -1), keepdim=True)
        reference = owner.global_reference.project(reference)
        if self.mode == 'bilinear':
            reference = F.interpolate(reference, size=value.shape[-2:], mode=self.mode,
                                      align_corners=False)
        elif self.mode == 'nearest':
            reference = F.interpolate(reference, size=value.shape[-2:], mode=self.mode)
        else:
            raise ValueError(self.mode)
        value = value + reference
        if not old.skip_tail:
            value = owner.tail(value)
        return old.output(value)


def load_candidate(scene, name, run_dir, device='cuda'):
    model = load_board(scene, 'rows32', run_dir, device=device)
    if name == 'rows32':
        return model
    if name.startswith('keep'):
        projected = name.endswith('_projected')
        trained = name.endswith('_trained') or projected
        stem = name[:-10] if projected else name[:-8] if trained else name
        keep = tuple(int(v) for v in stem[4:])
        old = model.core.model.global_reference
        if len(old.pyramid) != 3:
            raise ValueError('Pyramid ablation only applies to ordinary model')
        reference = SelectPyramid(old, keep).to(device=device).eval()
        if trained:
            checkpoint_dir = Path(os.environ.get('RAWIR_V11_CHECKPOINTS', str(Path(run_dir).parent / 'SS928-BOARD-V11-20260928')))
            suffix = '_projected' if projected else ''
            checkpoint = torch.load(checkpoint_dir / f'{scene}_{stem}{suffix}_reference.pt',
                                    map_location='cpu', weights_only=True)
            reference.load_state_dict(checkpoint['reference'], strict=True)
        model.core.model.global_reference = reference
        return model
    if name in ('nearest_reference', 'bilinear_wrapper'):
        mode = 'nearest' if name == 'nearest_reference' else 'bilinear'
        return ResizeReferenceSystem(model, mode).to(device=device).eval()
    if name.startswith('shallow'):
        trained = name.endswith('_trained')
        stem = name[:-8] if trained else name
        kernel = int(stem.removeprefix('shallow'))
        if kernel not in (3, 5):
            raise ValueError(name)
        owner = model.core.model
        reference = ShallowEncoderReference(owner.global_reference, kernel).to(device=device).eval()
        if trained:
            checkpoint_dir = Path(os.environ.get('RAWIR_V11_CHECKPOINTS', str(Path(run_dir).parent / 'SS928-BOARD-V11-20260928')))
            checkpoint = torch.load(checkpoint_dir / f'{scene}_{stem}_reference.pt',
                                    map_location='cpu', weights_only=True)
            reference.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        owner.global_reference = reference
        return model
    raise ValueError(name)
