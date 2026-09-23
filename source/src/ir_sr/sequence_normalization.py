"""Offline per-segment RAW calibration; no target images or learned parameters."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def segment_bounds(raw):
    """Frozen weather-v6 cut heuristic; full segment calibration uses future RAW."""
    cuts=[0];last=None
    for k,x in enumerate(raw):
        a=cv2.resize(x.astype(np.float32),(160,128),interpolation=cv2.INTER_AREA)
        if last is not None:
            d=a-last;span=float(np.percentile(a,99)-np.percentile(a,1))
            if abs(float(np.median(d)))>20 or np.mean(abs(d)>max(5.,.2*span))>.6:cuts.append(k)
        last=a
    return list(zip(cuts,cuts[1:]+[len(raw)]))


def calibrate(reference, low=.1, high=99.9, minimum_span=20.):
    if reference.ndim!=2 or not np.isfinite(reference).all():raise ValueError('Finite 2-D reference required')
    a,b=map(float,np.percentile(reference,[low,high]))
    if b-a<minimum_span:
        center=(a+b)/2;a=center-minimum_span/2;b=center+minimum_span/2
    return {'offset':a,'scale':b-a,'upper':b}


def calibrate_sequence(raw):
    """RAW-only calibration; caller must slice allowed ROI before this function."""
    if raw.ndim != 3 or not len(raw):
        raise ValueError('Expected nonempty T,H,W RAW')
    rows = []; segments = []
    for begin, end in segment_bounds(raw):
        source = raw[begin:end]
        centers = np.median(source[:, ::8, ::8], axis=(1, 2)).astype(np.float32)
        sampled = source[:, ::4, ::4].astype(np.float32)
        reference4 = sampled.mean(0) - centers.mean()
        common = np.clip(sampled - centers[:, None, None] - reference4, -4, 4).mean((1, 2), dtype=np.float64)
        bounds = calibrate(source.mean(0, dtype=np.float32))
        low = bounds['offset'] - float(centers.mean()) - float(common.mean())
        anchor = float(np.median(centers))
        # Frozen denoised targets already contain +anchor, not +current center.
        for center, residual in zip(centers, common):
            rows.append({'offset': float(center) + float(residual) + low,
                         'middle_offset': anchor + low, 'scale': bounds['scale']})
        segments.append({'begin': begin, 'end': end, 'center_anchor': anchor,
                         'corrected_low': low, 'scale': bounds['scale']})
    return rows, segments


class SequenceNormalization:
    def __init__(self, path, expected_sha256):
        if not expected_sha256 or digest(path) != expected_sha256:
            raise ValueError('Sequence normalization index identity mismatch')
        self.sha256 = expected_sha256
        self.index = json.loads(Path(path).read_text())
        if self.index['status'] != 'complete' or self.index['kind'] != 'raw_frame_center_segment_v1':
            raise ValueError('Unready normalization index')
        self.by_key = {r['key']: r for r in self.index['sequences']}
        if len(self.by_key) != len(self.index['sequences']):
            raise ValueError('Duplicate sequence calibration')
        for row in self.by_key.values():
            if len(row['parameters']) != row['frames']:
                raise ValueError('Calibration does not cover sequence')
            for p in row['parameters']:
                if not np.isfinite([p['offset'], p['middle_offset'], p['scale']]).all() or p['scale'] <= 0:
                    raise ValueError('Invalid calibration parameters')

    def for_record(self, record, frame_id=None):
        row = self.by_key[record['domain'] + '/' + record['sequence_id']]
        if row['raw_path'] != record['raw']['path']:
            raise ValueError('Calibration/source mismatch')
        roi = record.get('train_roi_tlhw', [0, 0, 1024, 1280])
        if row['roi_tlhw'] != roi or row['split'] != record['split']:
            raise ValueError('Calibration region/split mismatch')
        frame = record['frame_id'] if frame_id is None else frame_id
        if not 0 <= frame < row['frames']:
            raise ValueError('Frame outside calibrated sequence')
        return row['parameters'][frame]

    def validate_sources(self, root):
        for row in self.by_key.values():
            if digest(Path(root) / row['raw_path']) != row['raw_sha256']:
                raise ValueError('Calibration RAW source changed')
