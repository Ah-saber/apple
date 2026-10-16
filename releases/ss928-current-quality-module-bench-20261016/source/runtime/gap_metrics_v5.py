"""Full-field metric with GT-confirmed departure and arrival tracks."""
import cv2
from gap_metrics_v4 import GapMetrics as AppearanceMetric
from gt_tracks_v5 import GTTracks

class GapMetrics(AppearanceMetric):
    def __init__(self,crop,methods):
        super().__init__(crop,methods)
        self.tracker=GTTracks()
        y,x,h,w=self.crop
        self.tracker.origin=(y+3,x+3)

    def update(self,frame,gt,predictions):
        y,x,h,w=self.crop
        self.tracker.images[frame]=cv2.GaussianBlur(gt[y+3:y+h-3,x+3:x+w-3],(0,0),.7)
        self.tracker.noise=self.noise
        super().update(frame,gt,predictions)

