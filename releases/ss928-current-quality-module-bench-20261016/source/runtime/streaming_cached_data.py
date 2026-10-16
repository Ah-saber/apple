"""Correct only the new causal frame and retain K corrected frames."""
from collections import deque
import time
import numpy as np
import torch
from streaming_data import GuidanceView as BaseView, InputModel
from protected_data import correct_frame

class GuidanceView(BaseView):
    def __init__(self, source, indices):
        super().__init__(source,indices)
        self.corrected_streams = {}

    def correction(self,value,reference,pipeline):
        return correct_frame(value,reference,pipeline != 'streamCachedPlain')

    def inputs(self, pipeline):
        if not pipeline.startswith('streamCached'):
            return super().inputs(pipeline)
        if pipeline in self.cache:
            return self.cache[pipeline]
        from ir_sr.sequence_normalization import regional_key
        started = time.perf_counter()
        row = self.current_record
        norm = self.source.base.sequence_normalization
        name = regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
        frame = int(row['frame_id'])
        begin = next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin']<=frame<s['end'])
        delay = 0 if pipeline.endswith('Past') else 2
        ident = (name,begin,tuple(self.current_crop),pipeline)
        state = self.corrected_streams.get(ident)
        if state is None or state['frame'] > frame:
            state = dict(frame=begin-1, corrected=deque(maxlen=len(self.indices)), pending=deque(), background=None)
            self.corrected_streams[ident] = state
        start = state['frame']+1
        top,left,h,w = self.current_crop
        for f in range(start,frame+1):
            rr = dict(row,frame_id=f)
            if self.source.split == 'train' and (row['domain'],row['sequence_id'],f) not in self.source.train_ids:
                raise ValueError('Only training history is allowed')
            n = self.source.normalization_for(rr)
            raw = self.source.base._raw(rr)[top:top+h,left:left+w].astype(np.float32)
            value = (raw-n['offset'])/n['scale']
            if state['background'] is None:
                state['background'] = value.copy()
                for _ in range(delay):
                    state['pending'].append(value.copy())
            corrected = self.correction(value,state['background'],pipeline)
            state['corrected'].append(corrected)
            state['pending'].append(value)
            alpha = 2/11 if delay == 0 else 2/7
            state['background'] *= 1-alpha
            state['background'] += alpha*state['pending'].popleft()
            state['frame'] = f
        values = list(state['corrected'])
        values = [values[0]]*(len(self.indices)-len(values))+values
        value = torch.from_numpy(np.ascontiguousarray(np.stack(values)))
        self.cache[pipeline] = value
        self.preprocess_log.append(dict(pipeline=pipeline,frame=frame,advanced_frames=frame-start+1,
                                       elapsed_ms=(time.perf_counter()-started)*1000,
                                       corrections_per_new_frame=1))
        return value

