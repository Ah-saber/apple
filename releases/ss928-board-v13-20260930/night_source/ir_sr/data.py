from ir_sr.sequence_normalization import allowed_region
"""CPU dataset: preserve DN, crop aligned pairs, downsample RAW only."""
import json
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def area_downsample3(raw):
    if raw.ndim != 2 or any(d % 3 for d in raw.shape):
        raise ValueError('RAW patch must be 2-D with sides divisible by 3')
    a = raw.astype(np.float32)
    h, w = a.shape
    return a.reshape(h // 3, 3, w // 3, 3).mean(axis=(1, 3), dtype=np.float32)


def geometric_transform(array, operation):
    """Eight exact square symmetries; no interpolation or value changes."""
    if operation not in range(8):
        raise ValueError('Unknown geometric operation')
    out = np.rot90(array, operation % 4, axes=(-2, -1))
    if operation // 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def paired_patch(raw, gt, top, left, height, width, offset, scale):
    if raw.shape != gt.shape or height % 6 or width % 6 or scale <= 0:
        raise ValueError('Invalid paired geometry or normalization')
    if min(top, left) < 0 or top + height > raw.shape[0] or left + width > raw.shape[1]:
        raise ValueError('Crop outside frame')
    r = raw[top:top + height, left:left + width]
    g = gt[top:top + height, left:left + width]
    x = (area_downsample3(r) - offset) / scale
    y = g.astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None]), np.ascontiguousarray(y[None])


class RawDisplayDataset(Dataset):
    def __init__(self, root, split='train', seed=928, mode=None, max_open_sequences=8,
                 dataset_version='dataset_d0', scene_id=None, target_cache_root=None, scene_ids=None,
                 train_crop_hr=None, geometric_augmentation=False, middle_gt_root=None, sequence_normalization_index=None, sequence_normalization_sha256=None, global_reference=False, native_training=False, native_crop_sampling="center", absolute_raw_reference=False, reference_tone_supervision=False, reference_sensor_y=False, shared_raw_normalization_index=None, shared_raw_normalization_sha256=None, shared_raw_normalization_scene_ids=None, full_raw_reference_scene_ids=None):
        from ir_sr.sequence_normalization import SequenceNormalization
        self.sequence_normalization = SequenceNormalization(sequence_normalization_index, sequence_normalization_sha256) if sequence_normalization_index else None
        from ir_sr.shared_raw_normalization import SharedRawNormalization
        self.shared_raw_normalization = SharedRawNormalization(shared_raw_normalization_index,shared_raw_normalization_sha256) if shared_raw_normalization_index else None
        self.shared_raw_normalization_scene_ids=set(shared_raw_normalization_scene_ids or [])
        self.full_raw_reference_scene_ids=set(full_raw_reference_scene_ids or [])
        if bool(self.shared_raw_normalization_scene_ids)!=bool(self.shared_raw_normalization):raise ValueError('Shared RAW calibration requires explicit scene selection')
        if not self.full_raw_reference_scene_ids<=self.shared_raw_normalization_scene_ids:raise ValueError('Full RAW reference requires consistent full RAW normalization')
        if self.full_raw_reference_scene_ids and (not global_reference or reference_tone_supervision):raise ValueError('Shared RAW reference requires RAW-only reference, no reference GT head')
        self.reference_sensor_y = reference_sensor_y
        if reference_sensor_y and not global_reference:raise ValueError("Sensor row requires reference")
        self.reference_tone_supervision = reference_tone_supervision
        if reference_tone_supervision and (split!='train' or not global_reference):
            raise ValueError('Reference GT is training-only and requires RAW reference')
        self.global_reference = global_reference
        self.absolute_raw_reference = absolute_raw_reference
        if absolute_raw_reference and not global_reference:
            raise ValueError("Absolute reference requires global_reference")
        self.native_crop_sampling=native_crop_sampling
        if native_crop_sampling not in ("center","uniform"):raise ValueError("Invalid native sampling")
        self.native_training = native_training
        if native_training and split != "train":raise ValueError("Native adaptation is training-only")
        self.root = Path(root)
        self.mode = mode or ('train' if split == 'train' else 'validation')
        self.seed, self.epoch = seed, 0
        self.cache = OrderedDict()
        self.corrected_cache = {}
        self.target_cache = OrderedDict()
        if split != 'train' and (geometric_augmentation or middle_gt_root):
            raise ValueError('Augmentation and middle targets are training-only')
        self.augmentation = geometric_augmentation
        self.middle_root = Path(middle_gt_root) if middle_gt_root else None
        self.middle_cache = OrderedDict()
        self.middle_index = {}
        if self.middle_root:
            middle_manifest = json.loads((self.middle_root / 'index.json').read_text())
            if middle_manifest['status'] != 'complete' or middle_manifest['dataset'] != dataset_version:
                raise ValueError('Middle GT is not ready for this dataset')
            self.middle_index = {r['key']: r for r in middle_manifest['sequences']}
        self.target_cache_root = Path(target_cache_root) if target_cache_root else None
        self.target_cache_index = {}
        if self.target_cache_root:
            manifest = json.loads((self.target_cache_root / 'cache_index.json').read_text())
            if manifest['dataset'] != dataset_version or manifest['status'] != 'complete':
                raise ValueError('Target cache identity/completion mismatch')
            self.target_cache_index = {r['key']: r for r in manifest['sequences']}
        self.max_open = max_open_sequences
        if dataset_version not in ('dataset_d0', 'dataset_d1'):
            raise ValueError('Unknown dataset version')
        self.dataset_version = dataset_version
        recipe_name, split_name = ('preprocess_v1', 'split_s0') if dataset_version == 'dataset_d0' else ('preprocess_v2', 'split_s1')
        self.recipe = json.loads((self.root / f'recipes/{recipe_name}.json').read_text())
        crop = self.recipe['train_crop_hr'] if train_crop_hr is None else train_crop_hr
        crop = [crop, crop] if type(crop) is int else crop
        if (not isinstance(crop, (list, tuple)) or len(crop) != 2 or
                any(type(d) is not int or d <= 0 or d % 6 for d in crop)):
            raise ValueError('Training HR crop must contain two positive multiples of 6')
        if train_crop_hr is not None and split != 'train':
            raise ValueError('Training crop override is forbidden for evaluation splits')
        self.train_crop_hr = tuple(crop)
        if self.augmentation and crop[0] != crop[1]:
            raise ValueError('Eight-way rotations require square training crops')
        ids = (self.root / f'manifests/{split_name}/{split}.txt').read_text().splitlines()
        records = [json.loads(line) for line in (self.root / f'manifests/{dataset_version}/pairs.jsonl').read_text().splitlines()]
        by_id = {r['sample_id']: r for r in records}
        if len(by_id) != len(records) or len(ids) != len(set(ids)):
            raise ValueError('Duplicate samples')
        self.records = [by_id[i] for i in ids]
        self.crop_indices = list(range(len(self.records)))
        if scene_ids is not None:
            if dataset_version != 'dataset_d1' or scene_id is not None:
                raise ValueError('scene_ids requires D1 and cannot be combined with scene_id')
            if not scene_ids or len(set(scene_ids)) != len(scene_ids):
                raise ValueError('scene_ids must be nonempty and unique')
            present = {r['scene_id'] for r in self.records}
            if set(scene_ids) - present:
                raise ValueError('Requested scene absent from this split')
            selected = [(i, r) for i, r in enumerate(self.records) if r['scene_id'] in scene_ids]
            self.crop_indices = [i for i, _ in selected]
            self.records = [r for _, r in selected]
        if scene_id is not None:
            self.records = [r for r in self.records if r.get('scene_id') == scene_id]
            self.crop_indices = list(range(len(self.records)))
            if not self.records:
                raise ValueError('Scene absent from this split')
        if dataset_version == 'dataset_d1' and self.mode != ('train' if split == 'train' else 'validation'):
            raise ValueError('D1 mode cannot override split region restrictions')
        if self.mode == 'train':
            for record in self.records:
                region = record.get('train_roi_tlhw', [0, 0, 1024, 1280])
                if self.train_crop_hr[0] > region[2] or self.train_crop_hr[1] > region[3]:
                    raise ValueError('Training crop exceeds allowed region: ' + record['sample_id'])
                if self.middle_root:
                    entry = self.middle_index[record['domain'] + '/' + record['sequence_id']]
                    if (entry['roi_tlhw'] != region or record['frame_id'] not in entry['frame_ids'] or
                            record['sample_id'] not in entry['sample_ids'] or entry['source_raw_path'] != record['raw']['path']):
                        raise ValueError('Middle GT sample/region/source mismatch')

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        # Call before constructing each epoch's workers; persistent_workers=False.
        self.epoch = epoch

    def _raw(self, record):
        path = record['raw']['path']
        shared = record.get('scene_id') in self.shared_raw_normalization_scene_ids
        key = (path, record['frame_id'])
        if shared and key in self.corrected_cache:
            return self.corrected_cache[key]
        if path not in self.cache:
            self.cache[path] = np.load(self.root / path, mmap_mode='r', allow_pickle=False)
            if len(self.cache) > self.max_open:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        raw=self.cache[path][record['frame_id']]
        if shared:
            corrected = self.shared_raw_normalization.correct(record, raw, input_noise=True)
            corrected.flags.writeable = False
            self.corrected_cache[key] = corrected
            return corrected
        return raw

    def __getitem__(self, index):
        return self.item_with_geometry(index, index)

    def item_with_geometry(self, index, geometry_index):
        """Use a shared deterministic crop and transform for temporal frame pairs."""
        r = self.records[index]
        raw = self._raw(r)
        if self.target_cache_root:
            key = r['domain'] + '/' + r['sequence_id']
            record = self.target_cache_index[key]
            if record['source_target_hashes'][r['frame_id']] != r['target']['sha256']:
                raise ValueError('Target cache source identity mismatch')
            if key not in self.target_cache:
                value = np.load(self.target_cache_root / record['path'], mmap_mode='r', allow_pickle=False)
                if value.dtype != np.uint8 or list(value.shape) != record['shape']:
                    raise ValueError('Target cache shape or dtype mismatch')
                self.target_cache[key] = value
                if len(self.target_cache) > self.max_open:
                    self.target_cache.popitem(last=False)
            self.target_cache.move_to_end(key)
            gt = self.target_cache[key][r['frame_id']]
        else:
            with Image.open(self.root / r['target']['path']) as image:
                if image.mode != 'L':
                    raise ValueError('Target must be grayscale uint8')
                gt = np.array(image)
        if self.mode == 'train':
            height, width = self.train_crop_hr
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.crop_indices[geometry_index]]))
            rt, rl, rh, rw = r.get('train_roi_tlhw', [0, 0, *raw.shape])
            if rh < height or rw < width or min(rt, rl) < 0 or rt + rh > raw.shape[0] or rl + rw > raw.shape[1]:
                raise ValueError('Invalid training region')
            top = rt + int(rng.integers(rh - height + 1))
            left = rl + int(rng.integers(rw - width + 1))
        elif self.mode == 'validation':
            top, left, height, width = r.get('eval_crop_tlhw', self.recipe['validation_crop_tlhw'])
        else:
            raise ValueError('mode must be train or validation')
        norm = self.normalization_for(r)
        x, y = paired_patch(raw, gt, top, left, height, width, norm['offset'], norm['scale'])
        middle = None
        if self.middle_root:
            key = r['domain'] + '/' + r['sequence_id']
            entry = self.middle_index[key]
            if key not in self.middle_cache:
                value = np.load(self.middle_root / entry['path'], mmap_mode='r', allow_pickle=False)
                if value.dtype != np.float32 or list(value.shape) != entry['shape']:
                    raise ValueError('Middle GT shape/dtype mismatch')
                self.middle_cache[key] = value
                if len(self.middle_cache) > self.max_open:
                    self.middle_cache.popitem(last=False)
            self.middle_cache.move_to_end(key)
            rt, rl, _, _ = entry['roi_tlhw']
            frame = self.middle_cache[key][entry['frame_ids'].index(r['frame_id'])]
            if r.get('scene_id') in self.shared_raw_normalization_scene_ids:
                frame=self.shared_raw_normalization.correct(r,frame,entry['roi_tlhw'])
            patch = frame[top-rt:top-rt+height, left-rl:left-rl+width]
            if patch.shape != (height, width) or not np.isfinite(patch).all():
                raise ValueError('Invalid middle GT patch')
            middle = ((area_downsample3(patch) - norm.get('middle_offset', norm['offset'])) / norm['scale'])[None]
        operation = 0
        if self.augmentation:
            aug_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.crop_indices[geometry_index], 8107]))
            operation = int(aug_rng.integers(8))
            x, y = geometric_transform(x, operation), geometric_transform(y, operation)
            if middle is not None:
                middle = geometric_transform(middle, operation)
        item = {'raw': torch.from_numpy(x), 'gt': torch.from_numpy(y),
                'sample_id': r['sample_id'], 'crop_tlhw': (top, left, height, width)}
        if middle is not None:
            item['middle_raw'] = torch.from_numpy(np.ascontiguousarray(middle))
        if self.global_reference:
            context, box = self.context_for(r, (top, left, height, width), operation)
            item['context'], item['context_box'] = context, box
            if self.reference_tone_supervision:
                at,al,ah,aw=allowed_region(r)
                thumb_gt=cv2.resize(gt[at:at+ah,al:al+aw].astype(np.float32)/255.,(64,64),interpolation=cv2.INTER_AREA)
                item['reference_gt']=torch.from_numpy(geometric_transform(thumb_gt[None],operation))
        if self.native_training:
            # Same sample and augmentation as the old branch; native 256x256 at its center.
            nh,nw=height//3,width//3;nt,nl=top+(height-nh)//2,left+(width-nw)//2
            if self.native_crop_sampling=='uniform':
                nr=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,self.crop_indices[geometry_index],257]))
                at,al,ah,aw=allowed_region(r)
                nt=at+int(nr.integers(ah-nh+1));nl=al+int(nr.integers(aw-nw+1))
            native_raw=((raw[nt:nt+nh,nl:nl+nw].astype(np.float32)-norm['offset'])/norm['scale'])[None]
            native_gt=(gt[nt:nt+nh,nl:nl+nw].astype(np.float32)/255.)[None]
            item['raw_native']=torch.from_numpy(geometric_transform(native_raw,operation))
            item['gt_native']=torch.from_numpy(geometric_transform(native_gt,operation))
            item['native_crop_tlhw']=(nt,nl,nh,nw)
            if middle is not None:
                native_middle=((frame[nt-rt:nt-rt+nh,nl-rl:nl-rl+nw]-norm.get('middle_offset',norm['offset']))/norm['scale'])[None]
                item['middle_raw_native']=torch.from_numpy(geometric_transform(native_middle,operation))
            if self.global_reference:
                nc,nb=self.context_for(r,(nt,nl,nh,nw),operation)
                if not torch.equal(nc,item['context']):raise ValueError('Reference thumbnail changed across training scales')
                item['context_box_native']=nb
        item['raw_loss_multiplier'] = np.float32(norm['scale'] / self.recipe['raw_normalization']['scale'])
        item['normalization_offset'] = np.float32(norm['offset'])
        item['normalization_scale'] = np.float32(norm['scale'])
        item['middle_normalization_offset'] = np.float32(norm.get('middle_offset', norm['offset']))
        item['augmentation_id'] = operation
        if self.dataset_version == 'dataset_d1':
            item.update(scene_id=r['scene_id'], evaluation_scope=r['evaluation_scope'],
                        source_sample_id=r['source_sample_id'])
        return item

    def context_for(self, record, crop_tlhw=None, operation=0):
        raw = self._raw(record)
        default_crop=allowed_region(record)
        rt, rl, rh, rw = default_crop
        if record.get('scene_id') in getattr(self,'full_raw_reference_scene_ids',set()):
            rt,rl,rh,rw=0,0,*raw.shape
        norm = self.normalization_for(record)
        # RAW context scope is explicit; GT cropping continues to use the original allowed region.
        thumb = cv2.resize(raw[rt:rt+rh, rl:rl+rw].astype(np.float32), (64,64), interpolation=cv2.INTER_AREA)
        original_thumb = thumb
        thumb = (thumb-norm['offset'])/norm['scale']
        reference = thumb[None]
        if self.absolute_raw_reference:
            fixed = self.recipe['raw_normalization']
            absolute = (original_thumb-fixed['offset'])/fixed['scale']
            reference = np.stack((thumb, absolute), axis=0)
        if getattr(self,"reference_sensor_y",False):
            # Absolute sensor row, transformed jointly with RAW. No held-out intensity read.
            yy=(rt+(np.arange(64,dtype=np.float32)+.5)*rh/64)/raw.shape[0]*2-1
            row=np.broadcast_to(yy[:,None],(64,64))
            reference=np.concatenate((reference,row[None]),axis=0)
        top,left,h,w = crop_tlhw or default_crop
        box = np.array([(top-rt)/rh, (left-rl)/rw, (top+h-rt)/rh, (left+w-rl)/rw], dtype=np.float32)
        if np.any(box < 0) or np.any(box > 1) or operation not in range(8):
            raise ValueError('Invalid context crop/augmentation')
        for _ in range(operation % 4):
            y0,x0,y1,x1 = box
            box = np.array([1-x1,y0,1-x0,y1], dtype=np.float32)
        if operation // 4:
            y0,x0,y1,x1 = box
            box = np.array([y0,1-x1,y1,1-x0], dtype=np.float32)
        return torch.from_numpy(geometric_transform(reference, operation)), torch.from_numpy(box)

    def normalization_for(self, record, frame_id=None):
        if record.get("scene_id") in getattr(self,"shared_raw_normalization_scene_ids",set()):
            return self.shared_raw_normalization.for_record(record,frame_id)
        if self.sequence_normalization:
            return self.sequence_normalization.for_record(record, frame_id)
        return self.recipe['raw_normalization']

    def full_raw(self, index):
        r = self.records[index]
        if self.dataset_version == 'dataset_d1' and (r['evaluation_scope'] == 'spatial_development' or
                                                     r.get('train_roi_tlhw', [0, 0, 1024, 1280]) != [0, 0, 1024, 1280]):
            raise ValueError('Spatial development must infer on the cropped region; full-frame context is forbidden')
        n = self.normalization_for(r)
        raw = (self._raw(r).astype(np.float32) - n['offset']) / n['scale']
        return torch.from_numpy(np.ascontiguousarray(raw[None]))

    def balanced_sampler(self, seed=928):
        balance_key = 'scene_id' if self.dataset_version == 'dataset_d1' else 'domain'
        counts = Counter((r[balance_key], r['sequence_id']) for r in self.records)
        nseq = Counter(d for d, _ in counts)
        weights = [1.0 / (nseq[r[balance_key]] * counts[r[balance_key], r['sequence_id']]) for r in self.records]
        return WeightedRandomSampler(weights, len(weights), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
