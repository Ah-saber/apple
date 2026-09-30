"""Training-only consecutive frames, common geometry and GT-relative temporal loss."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from ir_sr.sequence_normalization import regional_key, allowed_region


def pair_identity(record):
    return (record['scene_id'], record['domain'], record['sequence_id'],
            record['raw']['path'], record['split'], tuple(allowed_region(record)))


class PairedFrames(Dataset):
    def __init__(self, base):
        self.base = base
        self.records = base.records
        if any(r['split'] != 'train' for r in self.records):
            raise ValueError('Temporal targets must come from training records only')
        lookup = {(pair_identity(r), r['frame_id']): i for i, r in enumerate(self.records)}
        if len(lookup) != len(self.records):
            raise ValueError('Duplicate temporal identity')
        self.neighbors = []
        for i, r in enumerate(self.records):
            candidates = [lookup.get((pair_identity(r), r['frame_id'] + delta)) for delta in (1, -1)]
            candidates = [j for j in candidates if j is not None and self.same_segment(r, self.records[j])]
            if not candidates:
                raise ValueError('No adjacent training frame in same calibration segment: '+r['sample_id'])
            self.neighbors.append(candidates[0])

    def same_segment(self, a, b):
        norm = self.base.sequence_normalization
        if norm is None:
            raise ValueError('Frozen temporal cut metadata required')
        key = regional_key(a) if norm.regional else a['domain']+'/'+a['sequence_id']
        return any(seg['begin'] <= min(a['frame_id'], b['frame_id']) and
                   max(a['frame_id'], b['frame_id']) < seg['end']
                   for seg in norm.by_key[key]['segments'])

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def balanced_sampler(self, seed):
        return self.base.balanced_sampler(seed)

    def __getitem__(self, index):
        a = self.base.item_with_geometry(index, index)
        b = self.base.item_with_geometry(self.neighbors[index], index)
        for key in ('crop_tlhw', 'native_crop_tlhw', 'augmentation_id'):
            if key in a and a[key] != b[key]:
                raise ValueError('Pair geometry differs')
        out = {}
        for key, value in a.items():
            if isinstance(value, torch.Tensor):
                out[key] = torch.stack((value, b[key]))
            elif key in ('raw_loss_multiplier', 'augmentation_id'):
                out[key] = torch.tensor([value, b[key]])
            else:
                out[key] = value
        return out

    def manifest(self):
        return {'kind': 'adjacent_train_frames_same_segment_same_geometry',
                'pairs': [[r['sample_id'], self.records[j]['sample_id']]
                          for r, j in zip(self.records, self.neighbors)]}


def flatten_pairs(batch):
    out = dict(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and (key.startswith(('raw', 'gt', 'middle_raw', 'context', 'reference_gt')) or key == 'augmentation_id'):
            if value.ndim < 2 or value.shape[1] != 2:
                raise ValueError('Pair dimension missing: '+key)
            out[key] = value.flatten(0, 1)
    out['scene_id'] = [scene for scene in batch['scene_id'] for _ in range(2)]
    return out


def residual_temporal_loss(prediction, target):
    """Subtract target temporal change; do not penalize correct motion/brightness change."""
    if prediction.shape != target.shape or len(prediction) % 2:
        raise ValueError('Expected interleaved aligned frame pairs')
    residual = prediction.float() - target.float()
    return (residual[0::2] - residual[1::2]).abs().mean()

def static_temporal_mask(target, change_gray=2, gradient_gray=5, exclusion_radius=2):
    """Select GT-stable, flat, unsaturated pixels away from detected motion."""
    if target.ndim != 4 or len(target) % 2 or target.shape[1] != 1:
        raise ValueError('Expected interleaved one-channel aligned GT pairs')
    if change_gray <= 0 or gradient_gray <= 0 or exclusion_radius < 0:
        raise ValueError('Invalid static-mask thresholds')
    pair = target.float().reshape(-1, 2, *target.shape[1:])
    first, second = pair[:, 0], pair[:, 1]
    change = (first - second).abs()
    mean = (first + second) * 0.5
    kernel_x = mean.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3) / 8
    kernel_y = kernel_x.transpose(-1, -2)
    with torch.autocast(device_type=target.device.type, enabled=False):
        gx = F.conv2d(F.pad(mean, (1, 1, 1, 1), mode='replicate'), kernel_x)
        gy = F.conv2d(F.pad(mean, (1, 1, 1, 1), mode='replicate'), kernel_y)
        diameter = 2 * exclusion_radius + 1
        near_motion = F.max_pool2d((change > change_gray / 255).float(), diameter,
                                   stride=1, padding=exclusion_radius) > 0
    return ((~near_motion) & (gx.square() + gy.square() <= (gradient_gray / 255) ** 2) &
            (mean > 5 / 255) & (mean < 250 / 255)).detach()


def static_residual_temporal_loss(prediction, target, change_gray=2, gradient_gray=5, exclusion_radius=2):
    """Penalize noise changes only where paired GT indicates a flat static region."""
    if prediction.shape != target.shape:
        raise ValueError('Prediction and GT shapes differ')
    mask = static_temporal_mask(target, change_gray, gradient_gray, exclusion_radius)
    error = (prediction.float() - target.float()).reshape(-1, 2, *target.shape[1:])
    difference = (error[:, 0] - error[:, 1]).abs()
    loss = (difference * mask).sum() / mask.sum().clamp_min(1)
    return loss, mask.float().mean()

def motion_residual_temporal_loss(prediction, target, change_gray=1.5, radius=1):
    """Match GT temporal change around moving structures in training pairs."""
    if prediction.shape != target.shape or prediction.ndim != 4 or len(target) % 2:
        raise ValueError('Expected interleaved aligned prediction and GT pairs')
    if change_gray <= 0 or radius < 0:
        raise ValueError('Invalid motion-mask settings')
    gt = target.float().reshape(-1, 2, *target.shape[1:])
    change = (gt[:, 0] - gt[:, 1]).abs()
    motion = (change > change_gray / 255).float()
    if radius:
        motion = F.max_pool2d(motion, 2*radius+1, stride=1, padding=radius)
    motion = motion.detach()
    pred = prediction.float().reshape(-1, 2, *prediction.shape[1:])
    delta_error = ((pred[:, 0]-pred[:, 1])-(gt[:, 0]-gt[:, 1])).abs()
    return (delta_error * motion).sum() / motion.sum().clamp_min(1), motion.mean()

def gt_motion_gate_loss(gate, target, low_gray=1., high_gray=3., positive_weight=5.):
    """Train a RAW-grid motion gate from aligned clean GT pair changes."""
    if gate.ndim != 4 or target.ndim != 4 or gate.shape[:2] != target.shape[:2] or len(gate) % 2:
        raise ValueError('Expected interleaved gate and GT pairs')
    if gate.shape[1] != 1 or target.shape[1] != 1 or low_gray <= 0 or high_gray <= low_gray or positive_weight <= 0:
        raise ValueError('Invalid motion-gate settings')
    scale = target.shape[-1] // gate.shape[-1]
    if scale not in (1,3) or target.shape[-2] != gate.shape[-2]*scale or target.shape[-1] != gate.shape[-1]*scale:
        raise ValueError('Expected equal or threefold GT resolution')
    pair = target.float().reshape(-1,2,*target.shape[1:])
    change = (pair[:,0]-pair[:,1]).abs()
    if scale == 3:
        change = F.max_pool2d(change,3,stride=3)
    positive = change > high_gray / 255
    negative = change < low_gray / 255
    label = positive.float().repeat_interleave(2,0)
    valid = (positive | negative).float().repeat_interleave(2,0)
    weight = torch.where(label > 0, positive_weight, 1.)
    with torch.autocast(device_type=target.device.type, enabled=False):
        bce = F.binary_cross_entropy(gate.float().clamp(1e-6,1-1e-6),label,reduction='none')
    return (bce*weight*valid).sum()/(weight*valid).sum().clamp_min(1), positive.float().mean(), valid.mean()
