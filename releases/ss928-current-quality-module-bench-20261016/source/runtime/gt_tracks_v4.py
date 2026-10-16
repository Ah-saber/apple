"""GT tracks require consistent moving residual appearance."""
from collections import defaultdict
import numpy as np
from gt_tracks import GTTracks as PreviousTracks

class GTTracks(PreviousTracks):
    def __init__(self):
        super().__init__()
        self.descriptors={}

    def valid(self,track):
        if len(track['items'])<5:return False
        points=np.asarray(track['points'])
        path=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        net=float(np.linalg.norm(points[-1]-points[0]))
        return net>=2 and net/max(path,1e-6)>=.5

    def update(self,frame,items):
        descriptors=[i.pop('_descriptor').flatten().astype(np.float32) for i in items]
        for d in descriptors:
            d-=d.mean()
            d/=max(float(np.linalg.norm(d)),1e-6)
        buckets=defaultdict(list)
        for ident in list(self.active):
            track=self.tracks[ident]
            if frame-track['last_frame']>2:
                del self.active[ident]
                continue
            py,px=track['points'][-1]
            buckets[(int(py//16),int(px//16))].append(ident)
        proposals=[]
        for i,item in enumerate(items):
            y,x=self.center(item)
            cy,cx=int(y//16),int(x//16)
            nearby=[ident for dy in [-1,0,1] for dx in [-1,0,1]
                    for ident in buckets[(cy+dy,cx+dx)]]
            for ident in nearby:
                track=self.tracks[ident]
                gap=frame-track['last_frame']
                if gap>2:
                    del self.active[ident];continue
                last=track['items'][-1]
                if item['polarity']!=last['polarity']:continue
                ratio=item['target_pixels']/last['target_pixels']
                if not .4<=ratio<=2.5:continue
                py,px=track['points'][-1]
                distance=float(np.hypot(y-py,x-px))
                if distance>6*gap:continue
                cosine=float(descriptors[i]@self.descriptors[ident])
                if cosine<.6:continue
                proposals.append((distance+3*(1-cosine),i,ident))
        assignments={};used=set()
        for _,i,ident in sorted(proposals):
            if i in assignments or ident in used:continue
            assignments[i]=ident;used.add(ident)
        confirmed=[]
        for i,item in enumerate(items):
            ident=assignments.get(i)
            if ident is None:
                ident=len(self.tracks)
                self.tracks.append({'items':[],'points':[],'last_frame':frame})
            track=self.tracks[ident]
            track['items'].append(item);track['points'].append(self.center(item));track['last_frame']=frame
            item['gt_track_id']=ident
            self.descriptors[ident]=descriptors[i]
            self.active[ident]=True
            if self.valid(track):confirmed.append(i)
        return confirmed

    def summary(self):
        result=super().summary()
        result['rule']='GT residual appearance cosine>=.6; max6px/frame; same polarity; gap<=2; >=5 observations; net>=2px; net/path>=.5'
        return result

