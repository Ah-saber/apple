"""Fast full-frame inference for current nine-frame checkpoints.

Uses the verified 2026-09-26 night runtime primitives. The reference branch keeps
the training model's context_box mapping, including the C32 spatial split.
"""
import math
import types

import torch
from torch.nn import functional as F

from collapse_output_shuffles import collapse_output_shuffles
from fused_nine import FusedNine
from trajectory_batched_stats import trajectory_features_batched_stats


class FullFrameFusedNine(FusedNine):
    def forward(self, stack, context=None, context_box=None):
        if context is None or context_box is None:
            raise ValueError('Context and context_box are required')
        model = self.model
        current = stack[:, -1:]
        gate_input = torch.cat((stack, model._trajectory_features(stack)), dim=1)
        joined = self.second(F.relu(self.first(self.half_input(gate_input)))).float()
        correction, logit = joined[:, :1], joined[:, 1:]
        denoised = current.float() + correction * (1. - torch.sigmoid(logit))
        features = model.body(model.head(model.down(self.half_input(denoised))))
        reference = model.global_reference(self.half_input(context), context_box,
                                            features.shape[-2:]).to(features.dtype)
        packed = model.upsample[0](model.tail(features + reference))
        return F.pixel_shuffle(packed.float(), math.prod(self.factors))


def build_fast_model(config, state_dict, inference_model):
    base = inference_model(config, state_dict).eval()
    if not (base.input_frames == 9 and base.motion_gate and base.trajectory_gate):
        raise ValueError('This runtime requires the nine-frame motion-gated architecture')
    base._trajectory_features = types.MethodType(trajectory_features_batched_stats, base)
    collapse_output_shuffles(base)
    return FullFrameFusedNine(base, raw_basis=True, channels_last=True,
                              dense_second=True).eval()
