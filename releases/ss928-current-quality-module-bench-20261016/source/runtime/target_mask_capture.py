"""Expose masks from the executed GT metric without changing its selection rules."""
import inspect
import textwrap
import cv2
import numpy as np
import gap_metrics_v4
from gap_metrics_v5 import GapMetrics

_source=textwrap.dedent(inspect.getsource(gap_metrics_v4.GapMetrics.update))
_old='frame_targets.append(item)'
if _source.count(_old)!=1: raise ValueError('GT mask capture seam differs')
_source=_source.replace(_old,"self.capture_mask(item,target,ring,top,left)\n        "+_old)
_namespace=dict(vars(gap_metrics_v4))
exec(compile(_source,'GT_mask_capture','exec'),_namespace)
_update=_namespace['update']

class CaptureMetric(GapMetrics):
    def __init__(self,crop,wanted=None):
        super().__init__(crop,['GT'])
        self.wanted=wanted; self.captured=[]

    def capture_mask(self,item,target,ring,top,left):
        if self.wanted is not None and (item['frame']!=self.wanted['frame'] or item['bbox_yxhw']!=self.wanted['bbox_yxhw']): return
        y,x,_,_=self.crop
        yy,xx=np.nonzero(target)
        ry,rx=np.nonzero(ring)
        self.captured.append(dict(item=item,target_yx=np.stack([yy+top+y+3,xx+left+x+3],1),ring_yx=np.stack([ry+top+y+3,rx+left+x+3],1)))

    def update(self,frame,gt,predictions=None):
        y,x,h,w=self.crop
        self.tracker.images[frame]=cv2.GaussianBlur(gt[y+3:y+h-3,x+3:x+w-3],(0,0),.7)
        self.tracker.noise=self.noise
        _update(self,frame,gt,{'GT':gt})
