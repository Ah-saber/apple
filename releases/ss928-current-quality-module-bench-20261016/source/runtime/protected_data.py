"""Causal compact-motion protection and construction from selected RAW frames."""
import cv2
import numpy as np


def compact_mask(residual):
    highpass = residual - cv2.GaussianBlur(residual, (0, 0), 2)
    center = float(np.median(highpass))
    sigma = max(1e-5, 1.4826 * float(np.median(np.abs(highpass - center))))
    seeds = (np.abs(highpass - center) >= 3 * sigma).astype(np.uint8)
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(seeds, 8)
    mask = np.zeros(seeds.shape, np.uint8)
    for ident in range(1, count):
        x, y, w, h, area = boxes[ident]
        if 2 <= area <= 256 and w <= 48 and h <= 64:
            patch = mask[y:y+h, x:x+w]
            patch[labels[y:y+h, x:x+w] == ident] = 1
    return cv2.dilate(mask, np.ones((7, 7), np.uint8)).astype(bool)


def correct_frame(frame, reference, protect=True):
    residual = frame - reference
    common = np.median(residual)
    row = np.median(residual - common, axis=1)[:, None]
    col = np.median(residual - common - row, axis=0)[None, :]
    local = residual - common - row - col
    broad = cv2.GaussianBlur(local, (0, 0), 4)
    if protect:
        weight = (~compact_mask(local)).astype(np.float32)
        denominator = cv2.GaussianBlur(weight, (0, 0), 4)
        numerator = cv2.GaussianBlur(local * weight, (0, 0), 4)
        broad = np.divide(numerator, denominator, out=broad.copy(), where=denominator >= .1)
    return np.ascontiguousarray(frame - common - row - col - .9 * broad, dtype=np.float32)


def correct_selected(stack, protect=True, reference='early'):
    if stack.ndim != 3 or len(stack) not in [3, 5, 7, 9]:
        raise ValueError('Expected a 3/5/7/9-frame normalized RAW stack')
    # For nine inputs this matches the existing six-frame background reference.
    if reference not in ['early', 'all']:
        raise ValueError('Unknown causal reference policy')
    field = np.median(stack if reference == 'all' else stack[:max(1, len(stack) - 3)], axis=0)
    output = [correct_frame(frame, field, protect) for frame in stack]
    return np.ascontiguousarray(np.stack(output), dtype=np.float32)


def normalized_selected(raw, parameters, frame, offsets, crop, begin=0, dynamic=True, protect=True):
    if offsets[-1] != 0 or any(v > 0 for v in offsets):
        raise ValueError('Only causal offsets ending at current frame are allowed')
    top, left, height, width = crop
    ids = [max(begin, frame + v) for v in offsets]
    # Read only these physical frames. No hidden nine-frame field estimate.
    images = []
    for f in ids:
        norm = parameters[f]
        image = raw[f, top:top+height, left:left+width].astype(np.float32)
        images.append((image - norm['offset']) / norm['scale'])
    stack = np.stack(images)
    return (correct_selected(stack, protect) if dynamic else stack), ids
