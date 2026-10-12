"""Retain corrected frames in a double ring; normalize and update background in native CPU loops."""
from collections import deque
import time
import ctypes
import numpy as np
import torch
from streaming_native_data import GuidanceView as BaseView,InputModel,LIB,F

LIB.normalize_frame.argtypes=[F,ctypes.c_float,ctypes.c_float,F,ctypes.c_int]
LIB.update_background.argtypes=[F,F,ctypes.c_float,ctypes.c_float,ctypes.c_int]
LIB.normalize_frame.restype=LIB.update_background.restype=None

class GuidanceView(BaseView):
    def _nine_without_field(self):
        from ir_sr.sequence_normalization import regional_key
        row=self.current_record; norm=self.source.base.sequence_normalization
        name=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
        frame=int(row['frame_id'])
        begin=next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin']<=frame<s['end'])
        ident=(name,begin,tuple(self.current_crop),'cached_original9')
        top,left,height,width=self.current_crop
        state=self.corrected_streams.get(ident)
        if state is None or state['frame']>frame:
            start=max(begin,frame-8)
            state=dict(frame=start-1,ring=np.empty((18,height,width),np.float32),position=-1)
            self.corrected_streams[ident]=state
        for f in range(state['frame']+1,frame+1):
            rr=dict(row,frame_id=f); n=self.source.normalization_for(rr)
            raw=np.ascontiguousarray(self.source.base._raw(rr)[top:top+height,left:left+width],dtype=np.float32)
            value=np.empty_like(raw)
            LIB.normalize_frame(raw,float(n['offset']),float(n['scale']),value,value.size)
            position=(state['position']+1)%9
            if state['position']<0: state['ring'][:]=value
            else:
                state['ring'][position]=value; state['ring'][position+9]=value
            state['position'],state['frame']=position,f
        position=state['position']
        return torch.from_numpy(state['ring'][position+1:position+10])

    def inputs(self,pipeline):
        if pipeline=='control9' and not self.source.dynamic_correction and self.source.split!='train' and len(self.indices)==9:
            if pipeline not in self.cache: self.cache[pipeline]=self._nine_without_field()
            return self.cache[pipeline]
        if not pipeline.startswith('streamCached'):
            return super().inputs(pipeline)
        if pipeline in self.cache:
            return self.cache[pipeline]
        from ir_sr.sequence_normalization import regional_key
        started=time.perf_counter()
        row=self.current_record; norm=self.source.base.sequence_normalization
        name=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
        frame=int(row['frame_id'])
        begin=next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin']<=frame<s['end'])
        delay=0 if pipeline.endswith('Past') else 2
        ident=(name,begin,tuple(self.current_crop),pipeline)
        state=self.corrected_streams.get(ident)
        top,left,height,width=self.current_crop; count=len(self.indices)
        if state is None or state['frame']>frame:
            state=dict(frame=begin-1,pending=deque(),background=None,ring=np.empty((2*count,height,width),np.float32),position=-1)
            self.corrected_streams[ident]=state
        start=state['frame']+1
        for f in range(start,frame+1):
            rr=dict(row,frame_id=f)
            if self.source.split=='train' and (row['domain'],row['sequence_id'],f) not in self.source.train_ids:
                raise ValueError('Only training history is allowed')
            n=self.source.normalization_for(rr)
            raw=np.ascontiguousarray(self.source.base._raw(rr)[top:top+height,left:left+width],dtype=np.float32)
            value=np.empty_like(raw)
            LIB.normalize_frame(raw,float(n['offset']),float(n['scale']),value,value.size)
            if state['background'] is None:
                state['background']=value.copy()
                for _ in range(delay): state['pending'].append(value.copy())
            corrected=self.correction(value,state['background'],pipeline)
            position=(state['position']+1)%count
            if state['position']<0:
                state['ring'][:]=corrected
            else:
                state['ring'][position]=corrected
                state['ring'][position+count]=corrected
            state['position']=position
            state['pending'].append(value)
            alpha=2/11 if delay==0 else 2/7
            LIB.update_background(state['background'],state['pending'].popleft(),1-alpha,alpha,value.size)
            state['frame']=f
        position=state['position']
        # A contiguous view is valid until the next advance of this same stream.
        value=torch.from_numpy(state['ring'][position+1:position+1+count])
        self.cache[pipeline]=value
        self.preprocess_log.append(dict(pipeline=pipeline,frame=frame,advanced_frames=frame-start+1,elapsed_ms=(time.perf_counter()-started)*1000,corrections_per_new_frame=1))
        return value
