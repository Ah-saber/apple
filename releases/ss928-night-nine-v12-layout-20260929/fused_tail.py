"""Compose the linear tail and four-phase projection into one 5x5 convolution."""
import torch
from torch import nn

from lowres_reference import tail_conv
from output_candidates import load_candidate as load_output


def composed_conv(tail, projection):
    first = tail_conv(tail)
    second = projection
    assert first.kernel_size == second.kernel_size == (3, 3)
    assert first.stride == second.stride == (1, 1)
    assert first.dilation == second.dilation == (1, 1)
    assert first.groups == second.groups == 1
    merged = nn.Conv2d(first.in_channels, second.out_channels, 5, padding=2,
        bias=True, device=first.weight.device, dtype=first.weight.dtype)
    a = first.weight.float()
    b = second.weight.float()
    kernel = torch.zeros_like(merged.weight, dtype=torch.float32)
    for y2 in range(3):
        for x2 in range(3):
            for y1 in range(3):
                for x1 in range(3):
                    kernel[:, :, y1 + y2, x1 + x2] += torch.einsum(
                        'om,mi->oi', b[:, :, y2, x2], a[:, :, y1, x1])
    bias = second.bias.float().clone() if second.bias is not None else torch.zeros(
        second.out_channels, device=first.weight.device)
    if first.bias is not None:
        bias += b.sum((-1, -2)) @ first.bias.float()
    with torch.no_grad():
        merged.weight.copy_(kernel.to(dtype=merged.weight.dtype))
        merged.bias.copy_(bias.to(dtype=merged.bias.dtype))
    return merged


def load_candidate(scene, layout, run_dir, device='cuda'):
    model = load_output(scene, layout, run_dir, device=device)
    owner = model.core.model
    model.output.conv = composed_conv(owner.tail, model.output.conv)
    owner.tail = nn.Identity()
    return model.eval()
