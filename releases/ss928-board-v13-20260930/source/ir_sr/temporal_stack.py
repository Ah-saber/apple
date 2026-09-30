"""Causal normalized RAW stack; only current GT, shared spatial geometry and D4."""
import numpy as np
import cv2
import torch
from ir_sr.data import area_downsample3,geometric_transform
from ir_sr.sequence_normalization import regional_key


def causal_dynamic_correct(stack):
    """Causal small-field correction in normalized RAW units; no GT or future frames."""
    if stack.ndim != 3 or stack.shape[0] != 9 or not np.isfinite(stack).all():
        raise ValueError('Expected finite nine-frame normalized RAW stack')
    reference = np.median(stack[:6], axis=0)
    corrected = []
    for frame in stack:
        residual = frame - reference
        common = np.median(residual)
        row = np.median(residual - common, axis=1)[:, None]
        col = np.median(residual - common - row, axis=0)[None, :]
        broad = cv2.GaussianBlur(residual - common - row - col, (0, 0), 4)
        corrected.append(frame - common - row - col - .9 * broad)
    return np.ascontiguousarray(np.stack(corrected), dtype=np.float32)


class TemporalStackDataset:
    def __init__(self,base,frames,split,dynamic_correction=False):
        if frames not in (3,9):raise ValueError('Expected causal three-frame or nine-frame input')
        self.base=base;self.frames=frames;self.split=split;self.dynamic_correction=bool(dynamic_correction)
        if self.dynamic_correction and frames != 9:raise ValueError("Dynamic correction requires nine frames")
        self.train_ids={(r['domain'],r['sequence_id'],r['frame_id']) for r in base.records}

    def __getattr__(self,name):
        return getattr(object.__getattribute__(self,'base'),name)

    def __len__(self):return len(self.base)

    def frame_ids(self,record,frame_id=None):
        frame=record['frame_id'] if frame_id is None else frame_id
        norm=self.base.sequence_normalization
        key=regional_key(record) if norm.regional else record['domain']+'/'+record['sequence_id']
        begin=next(s['begin'] for s in norm.by_key[key]['segments'] if s['begin']<=frame<s['end'])
        ids=[]
        for old in range(frame-self.frames+1,frame+1):
            old=max(begin,old)
            if self.split=='train' and (record['domain'],record['sequence_id'],old) not in self.train_ids:old=frame
            ids.append(old)
        return ids

    def normalized_stack(self,record,crop,operation=0,downsample=False):
        t,l,h,w=crop;frames=[]
        for frame in self.frame_ids(record):
            rr=dict(record,frame_id=frame);n=self.base.normalization_for(rr)
            raw=self.base._raw(rr)[t:t+h,l:l+w].astype(np.float32)
            if downsample:raw=area_downsample3(raw)
            frames.append(geometric_transform((raw-n['offset'])/n['scale'],operation))
        stack = np.stack(frames)
        if self.dynamic_correction:
            stack = causal_dynamic_correct(stack)
        return torch.from_numpy(np.ascontiguousarray(stack))

    def item_with_geometry(self,index,geometry_index):
        item=self.base.item_with_geometry(index,geometry_index);record=self.base.records[index]
        operation=item['augmentation_id'];item['raw']=self.normalized_stack(record,item['crop_tlhw'],operation,True)
        if 'raw_native' in item:item['raw_native']=self.normalized_stack(record,item['native_crop_tlhw'],operation,False)
        item['history_frame_ids']=self.frame_ids(record)
        return item

    def __getitem__(self,index):return self.item_with_geometry(index,index)
