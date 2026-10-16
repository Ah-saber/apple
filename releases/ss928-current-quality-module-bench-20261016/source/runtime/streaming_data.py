"""Causal streaming reference; K recent RAW inputs plus a persistent background state."""
from collections import deque
import time
import numpy as np
import torch
from data_guidance_view_v2 import GuidanceView as BaseView, InputModel
from protected_data import correct_frame

class GuidanceView(BaseView):
    def __init__(self, source, indices):
        super().__init__(source, indices)
        self.streams = {}
        self.preprocess_log = []

    def _stream(self, delay):
        from ir_sr.sequence_normalization import regional_key
        row = self.current_record
        norm = self.source.base.sequence_normalization
        name = regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
        frame = int(row['frame_id'])
        begin = next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin'] <= frame < s['end'])
        identity = (name, begin, tuple(self.current_crop), delay)
        state = self.streams.get(identity)
        if state is None or state['frame'] > frame:
            # A random-row evaluation replays only past RAW in the same segment.
            state = dict(frame=begin-1, raw=deque(maxlen=len(self.indices)), pending=deque(),
                         background=None, last_reference=None)
            self.streams[identity] = state
        top, left, h, w = self.current_crop
        start = state['frame']+1
        for f in range(start, frame+1):
            rr = dict(row, frame_id=f)
            if self.source.split == 'train' and (row['domain'],row['sequence_id'],f) not in self.source.train_ids:
                raise ValueError('Streaming training history must belong to training records')
            p = self.source.normalization_for(rr)
            raw = self.source.base._raw(rr)[top:top+h,left:left+w].astype(np.float32)
            value = (raw-p['offset'])/p['scale']
            if state['background'] is None:
                state['background'] = value.copy()
                for _ in range(delay):
                    state['pending'].append(value.copy())
            reference = state['background'].copy()
            state['raw'].append((f,value))
            state['pending'].append(value)
            old = state['pending'].popleft()
            alpha = 2/11 if delay == 0 else 2/7
            state['background'] *= (1-alpha)
            state['background'] += alpha*old
            state['last_reference'] = reference
            state['frame'] = f
        by_frame = dict(state['raw'])
        ids = [max(begin, frame-len(self.indices)+1+i) for i in range(len(self.indices))]
        values = [by_frame[f] for f in ids]
        return values, state['last_reference'], frame-start+1

    def inputs(self, pipeline):
        if not pipeline.startswith('stream'):
            return super().inputs(pipeline)
        if pipeline in self.cache:
            return self.cache[pipeline]
        started = time.perf_counter()
        delay = 0 if pipeline == 'streamPast' else 2
        values, reference, advanced = self._stream(delay)
        protect_all = pipeline == 'streamProtected'
        stack = [correct_frame(v, reference, protect_all or (pipeline == 'streamLast' and i == len(values)-1))
                 for i,v in enumerate(values)]
        value = torch.from_numpy(np.ascontiguousarray(np.stack(stack)))
        self.cache[pipeline] = value
        self.preprocess_log.append(dict(pipeline=pipeline, frame=int(self.current_record['frame_id']),
                                       advanced_frames=advanced,
                                       elapsed_ms=(time.perf_counter()-started)*1000))
        return value

