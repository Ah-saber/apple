"""Penalize residual target contrast at positions vacated in training GT."""
import torch
from torch.nn import functional as F


def departure_loss(prediction, current_gt, previous_gt, motion_mask, border):
    old = F.avg_pool2d(previous_gt, 3, 1, 1)
    current = F.avg_pool2d(current_gt, 3, 1, 1)
    old_contrast = (old-F.avg_pool2d(old, 7, 1, 3))*255
    current_contrast = (current-F.avg_pool2d(current, 7, 1, 3))*255
    polarity = old_contrast.sign()
    departed = (old_contrast.abs() >= 3) & (current_contrast*polarity <= old_contrast.abs()*.5)
    departed = departed[:, :, border:-border, border:-border].float()*motion_mask
    mask = F.max_pool2d(departed, 3, 1, 1)
    pixel_error = (prediction-current_gt)[:, :, border:-border, border:-border]*255
    pred_contrast = prediction-F.avg_pool2d(prediction, 7, 1, 3)
    gt_contrast = current_gt-F.avg_pool2d(current_gt, 7, 1, 3)
    contrast_error = (pred_contrast-gt_contrast)[:, :, border:-border, border:-border]*255
    old_sign = polarity[:, :, border:-border, border:-border]
    # Absolute pixel error prevents removing a real GT structure to hide a ghost.
    error = pixel_error.abs()+torch.relu(contrast_error*old_sign)
    return (error*mask).sum()/mask.sum().clamp_min(1), float(mask.detach().mean())
