"""Training-only small-region weights and confidence from existing middle RAW."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/data/zhangbenzhuang/huawei_sr')


def plain_raster(points):
    a = np.zeros((1024, 1280), np.int8)
    if len(points):
        a[points[:, 0], points[:, 1]] = points[:, 2]
    return a


def balanced_raster(points):
    a = plain_raster(points)
    result = np.zeros(a.shape, np.float32)
    for sign in [-1, 1]:
        count, labels, boxes, _ = cv2.connectedComponentsWithStats((a == sign).astype(np.uint8), 8)
        for ident in range(1, count):
            x, y, w, h, area = boxes[ident]
            patch = result[y:y+h, x:x+w]
            patch[labels[y:y+h, x:x+w] == ident] = sign / max(1, int(area))
    return result


def stratified_crops(saved, size, rng):
    ry, rx, rh, rw = saved['roi']
    pools = {'small': [], 'medium': [], 'large': [], 'ghost': []}
    for frame, pair in saved['labels'].items():
        if frame % 4 or frame == 0:
            continue
        for role in ['target', 'ghost']:
            a = plain_raster(pair[role])
            _, _, boxes, centers = cv2.connectedComponentsWithStats((a != 0).astype(np.uint8), 8)
            for box, center in zip(boxes[1:], centers[1:]):
                area = int(box[4])
                key = 'ghost' if role == 'ghost' else 'small' if area <= 9 else 'medium' if area <= 64 else 'large'
                x, y = center
                top = int(np.clip(round((y-size/2)/2)*2, (ry+1)//2*2, (ry+rh-size)//2*2))
                left = int(np.clip(round((x-size/2)/2)*2, (rx+1)//2*2, (rx+rw-size)//2*2))
                pools[key].append((frame, top, left))
    chosen = []
    for key, cap in [('small', 24), ('medium', 12), ('large', 8), ('ghost', 24)]:
        pool = list(dict.fromkeys(pools[key]))
        if pool:
            chosen.extend(pool[i] for i in rng.choice(len(pool), min(cap, len(pool)), replace=False))
    frames = [f for f in saved['labels'] if f % 4 == 0]
    for _ in range(16):
        frame = int(rng.choice(frames))
        top = int(rng.integers((ry+1)//2, (ry+rh-size)//2+1))*2
        left = int(rng.integers((rx+1)//2, (rx+rw-size)//2+1))*2
        chosen.append((frame, top, left))
    return list(dict.fromkeys(chosen))


class MiddleConfidence:
    def __init__(self, config, sample):
        self.middle, self.entry = None, None
        root = config.get('middle_gt_root')
        if not root:
            return
        index = json.loads((Path(root) / 'index.json').read_text())
        key = sample['domain'] + '/' + sample['sequence_id']
        entries = [r for r in index['sequences'] if r['key'] == key]
        if not entries:
            return
        self.entry = entries[0]
        self.middle = np.load(Path(root) / self.entry['path'], mmap_mode='r', allow_pickle=False)
        self.frame_index = {int(f): i for i, f in enumerate(self.entry['frame_ids'])}

    def crop(self, data, record, crop):
        top, left, h, w = crop
        if self.middle is None or record['frame_id'] not in self.frame_index:
            return torch.ones((1, 1, h, w), dtype=torch.float32)
        raw = data.base._raw(record)[top:top+h, left:left+w].astype(np.float32)
        middle = self.middle[self.frame_index[record['frame_id']]]
        if middle.shape == (1024, 1280):
            mid = middle[top:top+h, left:left+w]
        else:
            my, mx, mh, mw = self.entry['roi_tlhw']
            if not (my <= top and mx <= left and top+h <= my+mh and left+w <= mx+mw):
                raise ValueError('Training crop falls outside middle RAW ROI')
            mid = middle[top-my:top-my+h, left-mx:left-mx+w]
        if mid.shape != raw.shape:
            raise ValueError('Middle RAW crop shape mismatch')
        norm = data.normalization_for(record)
        residual = (raw-norm['offset'])/norm['scale'] - (mid.astype(np.float32)-norm.get('middle_offset', norm['offset']))/norm['scale']
        residual -= np.median(residual)
        sigma = max(1/norm['scale'], 1.4826 * float(np.median(np.abs(residual))))
        local = cv2.GaussianBlur(np.abs(residual), (0, 0), 2)
        confidence = np.clip(1/(1+(local/(3*sigma))**2), .25, 1).astype(np.float32)
        return torch.from_numpy(confidence)[None, None]


def balanced_objective(pred, gt, teacher, target, ghost, static_confidence):
    mask, depart = (target != 0).float(), (ghost != 0).float()
    e = (pred-gt)*255
    gt_local = gt-F.avg_pool2d(gt, 7, 1, 3)
    gap = (pred-F.avg_pool2d(pred, 7, 1, 3))-gt_local
    weak_target = (12/(gt_local.abs()*255).clamp_min(4)).clamp(1, 3)
    tw, dw = target.abs()*weak_target, ghost.abs()
    centered = e-F.avg_pool2d(e, 21, 1, 10)
    region = (slice(None), slice(None), slice(12, -12), slice(12, -12))

    def weighted(value, weight):
        value, weight = value[region], weight[region]
        return (value*weight).sum()/weight.sum().clamp_min(1e-8)

    pixels = weighted(e.abs(), tw)
    contrast = weighted(gap.abs()*255, tw)
    old = weighted(F.relu(centered*ghost.sign()), dw)
    old_abs = weighted(centered.abs(), dw)
    static = (1-F.max_pool2d(torch.maximum(mask, depart), 7, 1, 3)).clamp(0, 1)
    static *= static_confidence
    imitate = weighted((pred-teacher).abs()*255, static)
    teach_gap = (pred-F.avg_pool2d(pred, 7, 1, 3))-(teacher-F.avg_pool2d(teacher, 7, 1, 3))
    weak = weighted(teach_gap.abs()*255, static)
    pair = e.reshape(-1, 2, 1, e.shape[-2], e.shape[-1])
    temporal_weight = torch.maximum(tw[::2], tw[1::2])
    temporal = weighted((pair[:, 1]-pair[:, 0]).abs(), temporal_weight)
    loss = 4*pixels+6*contrast+8*old+2*old_abs+3*imitate+weak+2*temporal
    return loss, {'foreground_gray': float(pixels.detach()), 'gap_gray': float(contrast.detach()),
                  'ghost_gray': float(old.detach()), 'static_teacher_gray': float(imitate.detach()),
                  'middle_confidence_mean': float(static_confidence.mean().detach())}

