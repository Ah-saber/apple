"""Require GT appearance at the new location and departure from the old location."""
import cv2
import numpy as np
from gt_tracks_v4 import GTTracks as AppearanceTracks

class GTTracks(AppearanceTracks):
    def __init__(self):
        super().__init__()
        self.images={}
        self.origin=(0,0)
        self.noise=None

    def sample(self,image,point):
        y,x=point
        y=int(np.clip(round(y-self.origin[0]),1,image.shape[0]-2))
        x=int(np.clip(round(x-self.origin[1]),1,image.shape[1]-2))
        return float(image[y-1:y+2,x-1:x+2].mean()), (y,x)

    def valid(self,track):
        if not super().valid(track):return False
        first,last=track['items'][0],track['items'][-1]
        a,b=self.images[first['frame']],self.images[last['frame']]
        old,new=track['points'][0],track['points'][-1]
        ao,old_pixel=self.sample(a,old);bo,_=self.sample(b,old)
        an,new_pixel=self.sample(a,new);bn,_=self.sample(b,new)
        polarity=first['polarity']
        threshold=max(3.,.35*min(first['gt_gap_gray'],last['gt_gap_gray']))
        if self.noise is not None:
            threshold=max(threshold,1.5*float(max(self.noise[old_pixel],self.noise[new_pixel])))
        return polarity*(ao-bo)>=threshold and polarity*(bn-an)>=threshold

    def summary(self):
        result=super().summary()
        result['rule']+='; GT first-to-last departure and arrival >=max(3gray,.35local contrast,1.5GT MAD sigma)'
        return result

