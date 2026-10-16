"""Keep independent tiny guards; assign internal fragments to confirmed bodies."""
import cv2,numpy as np

def owner167(states,physical):
    owner=np.zeros_like(physical,dtype=np.uint8)
    for state in states:
        y,x,h,w=state['box'];points=cv2.findNonZero(state['component'].astype(np.uint8))
        if points is None or len(points)<3:continue
        hull=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(points),1)
        owner[y:y+h,x:x+w]|=hull
    return cv2.dilate(owner,np.ones((7,7),np.uint8))

def protection167(tracks,states,physical,gain):
    owner=owner167(states,physical)
    protect=np.zeros_like(owner);released=0;retained=0
    for y,x,h,w,polarity,component,history in tracks.weak.last_matches:
        selected=component>0
        strength=float(np.mean(np.abs(gain*physical[y:y+h,x:x+w][selected])))
        guarded=(polarity>0 and strength<=16 and (selected.sum()<=32 or max(h,w)<=12)) or (polarity<0 and selected.sum()<=9)
        if not guarded:continue
        contained=float(np.mean(owner[y:y+h,x:x+w][selected]))
        if contained>=.8:released+=1;continue
        protect[y:y+h,x:x+w]|=selected.astype(np.uint8);retained+=1
    return cv2.dilate(protect,np.ones((13,13),np.uint8)),dict(released=released,retained=retained)

