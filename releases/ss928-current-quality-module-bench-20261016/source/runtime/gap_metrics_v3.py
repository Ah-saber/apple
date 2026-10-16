"""GT-only moving target masks and local contrast over every valid GT pixel."""
from collections import deque
import cv2
import numpy as np
from gt_tracks import GTTracks


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)


def stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'p10': None, 'p95': None}
    a = np.asarray(values, np.float64)
    return {'count': len(values), 'mean': float(a.mean()),
            'p10': float(np.percentile(a, 10)), 'p95': float(np.percentile(a, 95))}


class GapMetrics:
    def __init__(self, crop, methods):
        self.crop = tuple(crop)
        self.methods = list(methods)
        self.previous_gt = None
        self.previous_predictions = None
        self.history = deque(maxlen=8)
        self.frames = []
        self.targets = []
        self.worst = None
        self.background = None
        self.noise = None
        self.tracker = GTTracks()

    def prepare_background(self, frames):
        y, x, h, w = self.crop
        values = np.stack([cv2.GaussianBlur(g[y+3:y+h-3, x+3:x+w-3].astype(np.float32), (0, 0), .7) for g in frames])
        initial = np.median(values, axis=0)
        for a in values:
            a -= np.median(a-initial)
        self.background = np.median(values, axis=0)
        self.noise = np.maximum(1., 1.4826*np.median(np.abs(values-self.background), axis=0)).astype(np.float32)

    def update(self, frame, gt, predictions):
        y, x, h, w = self.crop
        g = gt[y+3:y+h-3, x+3:x+w-3].astype(np.float32)
        ps = {name: image[y+3:y+h-3, x+3:x+w-3].astype(np.float32)
              for name, image in predictions.items()}
        result = {'frame': frame, 'methods': {}}
        for name, p in ps.items():
            e = p-g
            result['methods'][name] = {'psnr_db': float(-10*np.log10(max(float(np.mean(e*e))/255**2, 1e-12))),
                                      'mae_gray': float(np.abs(e).mean()), 'bias_gray': float(e.mean())}
        if self.previous_gt is None:
            self.previous_gt = g
            self.previous_predictions = ps
            self.frames.append(result)
            return
        dg = g-self.previous_gt
        dg -= np.median(dg)
        filtered_delta = cv2.GaussianBlur(dg, (0, 0), .7)
        residual = cv2.GaussianBlur(g, (0, 0), .7)-self.background
        residual -= np.median(residual)
        motion = np.abs(filtered_delta) >= np.maximum(3., 2*self.noise)
        weak_motion = np.abs(filtered_delta) >= np.maximum(1.5, self.noise)
        plausible = np.abs(residual) >= np.maximum(3., 2.5*self.noise)
        strong = np.abs(residual) >= np.maximum(5., 3*self.noise)
        seeds = plausible.astype(np.uint8)
        n, labels, boxes, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
        exclusion = dilate(plausible, 1)
        foreground = np.zeros(g.shape, np.int8)
        frame_targets = []
        frame_masks = []
        for label in range(1, n):
            bx, by, bw, bh, area = boxes[label]
            if not (2 <= area <= 256 and bw <= 48 and bh <= 64):
                continue
            top, left = max(0, by-8), max(0, bx-8)
            bottom, right = min(g.shape[0], by+bh+8), min(g.shape[1], bx+bw+8)
            component = labels[top:bottom, left:right] == label
            if int((component & weak_motion[top:bottom, left:right]).sum()) < 2 or int((component & strong[top:bottom, left:right]).sum()) < 2:
                continue
            ring = dilate(component, 6) & ~dilate(component, 2) & ~exclusion[top:bottom, left:right]
            if int(ring.sum()) < 12:
                continue
            local_gt = g[top:bottom, left:right]
            background = float(np.median(local_gt[ring]))
            # Exclude departure-only lobes: the current GT must still contain contrast.
            positive = component & (local_gt-background >= 3)
            negative = component & (local_gt-background <= -3)
            polarity = 1 if positive.sum() >= negative.sum() else -1
            target = positive if polarity == 1 else negative
            if int(target.sum()) < 2:
                continue
            # Recover the overlapping target interior, while retaining a local GT mask.
            target = dilate(target, 2) & (polarity*(local_gt-background) >= 3)
            if not (2 <= int(target.sum()) <= 256):
                continue
            gt_gap = polarity*(float(local_gt[target].mean())-background)
            gt_noise = float(np.std(local_gt[ring]))
            item = {'frame': frame, 'bbox_yxhw': [int(top+y+3), int(left+x+3), int(bottom-top), int(right-left)],
                    'target_pixels': int(target.sum()), 'polarity': polarity, 'gt_gap_gray': gt_gap,
                    'gt_background_std_gray': gt_noise, 'methods': {}}
            target_y, target_x = np.nonzero(target)
            item['centroid_yx'] = [float(target_y.mean()+top+y+3), float(target_x.mean()+left+x+3)]
            for name, p in ps.items():
                patch = p[top:bottom, left:right]
                gap = polarity*(float(patch[target].mean())-float(np.median(patch[ring])))
                noise = float(np.std(patch[ring]))
                item['methods'][name] = {'gap_gray': gap, 'gap_ratio_to_gt': gap/max(gt_gap, 1e-6),
                                         'gap_error_gray': abs(gap-gt_gap), 'background_std_gray': noise,
                                         'contrast_to_noise': gap/max(noise, 1.), 'polarity_inverted': float(gap < 0)}
            foreground[top:bottom, left:right][target] = polarity
            frame_targets.append(item)
            frame_masks.append((top, left, bottom, right, target, polarity))
        all_foreground = foreground.copy()
        foreground.fill(0)
        confirmed = self.tracker.update(frame, frame_targets)
        for i in confirmed:
            top, left, bottom, right, target, polarity = frame_masks[i]
            foreground[top:bottom, left:right][target] = polarity
            item = frame_targets[i]
            weakness = item['methods'][self.methods[-1]]['gap_ratio_to_gt']
            if self.worst is None or weakness < self.worst['ratio']:
                self.worst = {'ratio': weakness, 'record': item,
                              'images': {'GT': g[top:bottom, left:right].copy(), **{k: p[top:bottom, left:right].copy() for k, p in ps.items()}}}
        result['raw_region_count'] = len(frame_targets)
        result['confirmed_moving_target_count'] = len(confirmed)
        historical = np.zeros(g.shape, np.int8)
        for old in self.history:
            historical[old != 0] = old[old != 0]
        ghost = (historical != 0) & ~dilate(all_foreground != 0, 3) & ~dilate(motion, 2)
        static = ~dilate(motion, 3)
        if self.history:
            static &= ~dilate(historical != 0, 3)
        for name, p in ps.items():
            r = result['methods'][name]
            residual = p-g
            # Remove local residual bias, so a global brightness error does not count as a ghost.
            centered = residual-cv2.blur(residual, (21, 21), borderType=cv2.BORDER_REFLECT)
            r['old_position_pixels'] = int(ghost.sum())
            if ghost.any():
                r['old_position_error_gray'] = float(np.abs(centered[ghost]).mean())
                r['old_position_signed_ghost_gray'] = float(np.maximum(centered[ghost]*historical[ghost], 0).mean())
            if static.any():
                residual_change = (p-self.previous_predictions[name])-dg
                residual_change -= np.median(residual_change[static])
                r['static_temporal_error_gray'] = float(np.abs(residual_change[static]).mean())
            # Low-contrast spatial structures are checked separately from targets.
            local_contrast = g-cv2.blur(g, (7, 7), borderType=cv2.BORDER_REFLECT)
            weak = static & (np.abs(local_contrast) >= 1) & (np.abs(local_contrast) < 3)
            if weak.any():
                pred_contrast = p-cv2.blur(p, (7, 7), borderType=cv2.BORDER_REFLECT)
                r['weak_structure_error_gray'] = float(np.abs(pred_contrast-local_contrast)[weak].mean())
        self.history.append(foreground)
        self.targets.extend(frame_targets)
        self.frames.append(result)
        self.previous_gt = g
        self.previous_predictions = ps

    def report(self):
        selected_targets = self.tracker.selected()
        output = {'valid_crop_tlhw': list(self.crop), 'border_pixels': 3,
                  'selection': 'all valid GT pixels; GT frame differences only; both target polarities; no prediction selected masks',
                  'motion_threshold_gray': 'max(3, 2*GT residual MAD sigma)', 'weak_motion_threshold_gray': 'max(1.5, GT sigma)',
                  'GT_background': 'sequence median after Gaussian sigma .7 and per-frame drift removal; evaluation only',
                  'target_residual_threshold': 'max(3, 2.5*GT sigma); >=2 strong pixels above max(5, 3*GT sigma)',
                  'component_area_pixels': [2, 256], 'frames': self.frames, 'targets': selected_targets,
                  'raw_region_observations': len(self.targets), 'GT_tracks': self.tracker.summary(),
                  'ghost_selection': 'historical masks from online confirmed moving GT tracks only; eight-frame history; current foreground and GT motion excluded',
                  'target_observations': len(selected_targets), 'summary': {}}
        for name in self.methods:
            keys = set().union(*(f['methods'][name].keys() for f in self.frames))
            frame_summary = {k: stats([f['methods'][name][k] for f in self.frames if k in f['methods'][name]]) for k in keys}
            target_keys = list(selected_targets[0]['methods'][name]) if selected_targets else []
            target_summary = {k: stats([t['methods'][name][k] for t in selected_targets]) for k in target_keys}
            output['summary'][name] = {'frame_metrics': frame_summary, 'target_metrics': target_summary}
        return output

