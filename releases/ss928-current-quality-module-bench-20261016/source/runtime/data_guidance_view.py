"""Evaluate selected-frame preprocessing using the existing dataset frame contract."""
import numpy as np
import torch
from torch import nn
from protected_data import correct_selected


class GuidanceView:
    def __init__(self, source, indices):
        self.source, self.indices = source, list(indices)
        self.current_record = None
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def normalized_stack(self, record, crop, operation=0, downsample=False):
        self.current_record, self.current_crop = record, crop
        self.cache = {}
        value = self.source.normalized_stack(record, crop, operation, downsample)
        self.cache['control9'] = value
        self.cache['legacyK'] = value[self.indices].contiguous()
        return value

    def inputs(self, pipeline):
        if pipeline in self.cache:
            return self.cache[pipeline]
        record = self.current_record
        if record is None:
            raise RuntimeError('Set the current record through normalized_stack first')
        ids = self.source.frame_ids(record)
        selected = ids if pipeline == 'protected9' else [ids[i] for i in self.indices]
        top, left, h, w = self.current_crop
        values = []
        for f in selected:
            rr = dict(record, frame_id=f)
            norm = self.source.normalization_for(rr)
            raw = self.source.base._raw(rr)[top:top+h, left:left+w].astype(np.float32)
            values.append((raw - norm['offset']) / norm['scale'])
        stack = np.stack(values)
        if self.source.dynamic_correction:
            stack = correct_selected(stack, protect=pipeline.startswith('protected'))
        self.cache[pipeline] = torch.from_numpy(np.ascontiguousarray(stack))
        return self.cache[pipeline]


class InputModel(nn.Module):
    def __init__(self, board, view, pipeline, kind):
        super().__init__()
        self.board, self.view, self.pipeline, self.kind = board, view, pipeline, kind

    def native(self, raw, context, box):
        inputs = self.view.inputs(self.pipeline)[None].cuda()
        if not self.kind:
            inputs, context = inputs.half(), context.half()
        return self.board(inputs, context)[..., ::3, ::3].float() / 255
