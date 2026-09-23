"""Display-domain grayscale PSNR/SSIM; clipped float [0,1], HR border shave=3."""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def image_metrics(prediction, target, border=3):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
        raise ValueError('Metrics require matching one-image grayscale NCHW tensors')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite evaluation pixels')
    out_of_range = float(((prediction < 0) | (prediction > 1)).float().mean())
    p = prediction.float().clamp(0, 1)
    t = target.float()
    if border:
        p, t = p[:, :, border:-border, border:-border], t[:, :, border:-border, border:-border]
    if min(p.shape[-2:]) < 11:
        raise ValueError('SSIM requires at least 11 pixels after border cropping')
    mse = float((p.double() - t.double()).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-12))  # Exact identity capped at 120 dB for finite JSON.
    axis = torch.arange(-5, 6, device=p.device, dtype=torch.float64)
    kernel = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    p, t = p.double(), t.double()
    moments = torch.cat((p, t, p * p, t * t, p * t), dim=1)
    moments = F.conv2d(moments, kernel.reshape(1, 1, 1, 11).expand(5, 1, 1, 11), groups=5)
    moments = F.conv2d(moments, kernel.reshape(1, 1, 11, 1).expand(5, 1, 11, 1), groups=5)
    mu_p, mu_t, second_p, second_t, cross = moments.unbind(1)
    var_p, var_t = second_p - mu_p.square(), second_t - mu_t.square()
    covariance = cross - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) /
            ((mu_p.square() + mu_t.square() + 0.01 ** 2) * (var_p + var_t + 0.03 ** 2)))
    return {'psnr': psnr, 'ssim': float(ssim.mean()), 'mse': mse,
            'output_out_of_range_fraction': out_of_range}
