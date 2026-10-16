"""Maintain RAW-confirmed large bodies through brief seed-detection gaps."""
from collections import deque
import cv2,numpy as np

def overlap(a,b):
    y,x,h,w=a;yy,xx,hh,ww=b
    area=max(0,min(y+h,yy+hh)-max(y,yy))*max(0,min(x+w,xx+ww)-max(x,xx))
    return area/max(1,h*w+hh*ww-area)

class PersistentBodies154:
    def __init__(self,road):self.road=road;self.states=[];self.bank=deque(maxlen=4)
    def step(self,current,tracks,contrast,gain,protect):
        height,width=current.shape;states=[]
        for s in self.states:
            y,x,h,w=s['box'];radius=8;y0,x0=max(0,y-radius),max(0,x-radius);y1,x1=min(height,y+h+radius),min(width,x+w+radius)
            search=cv2.GaussianBlur(current[y0:y1,x0:x1],(0,0),.65)
            response=cv2.matchTemplate(search,s['template'],cv2.TM_CCOEFF_NORMED)
            yy,xx=np.mgrid[:response.shape[0],:response.shape[1]];dy=yy+y0-y;dx=xx+x0-x
            iy,ix=np.unravel_index(np.argmax(response-.001*(dy*dy+dx*dx)),response.shape)
            ny,nx=int(iy+y0),int(ix+x0);patch=current[ny:ny+h,nx:nx+w]
            energy=float(np.mean(np.abs(patch[s['component']])))
            confidence=float(response[iy,ix])
            if confidence<.7 or energy<max(.5,.25*s['energy']):continue
            strong=confidence>=.85 and energy>=.5*s['energy']
            if not strong and s['missed']>=8:continue
            s['box']=(ny,nx,h,w);s['missed']=0 if strong else s['missed']+1;states.append(s)
        observed=[]
        for r in tracks.medium.last_matches:
            y,x,h,w=r['y'],r['x'],r['h'],r['w'];component=r['component']
            if r['seeds']<128 or min(h,w)<24:continue
            energy=float(np.mean(np.abs(current[y:y+h,x:x+w][component])))
            if energy<.5:continue
            observed.append(dict(box=(y,x,h,w),component=component.copy(),missed=0,energy=energy))
        pairs=sorted([(overlap(r['box'],s['box']),ri,si) for ri,r in enumerate(observed) for si,s in enumerate(states)],key=lambda q:q[0],reverse=True)
        used_observed=set();used_states=set()
        for score,ri,si in pairs:
            if score<.25:break
            if ri in used_observed or si in used_states:continue
            states[si].update(observed[ri]);used_observed.add(ri);used_states.add(si)
        for ri,r in enumerate(observed):
            if ri in used_observed:continue
            s=dict(history=deque(maxlen=4),preserve_history=deque(maxlen=2));s.update(r);states.append(s)
            used_observed.add(ri);used_states.add(len(states)-1)
        mask=np.zeros_like(current);weight=np.zeros_like(current);total=np.zeros_like(current);count=np.zeros_like(current);records=[]
        for s in states:
            y,x,h,w=s['box'];component=s['component'];patch=current[y:y+h,x:x+w]
            selected=component&(gain*patch>4)
            signal=float(np.median(contrast[y:y+h,x:x+w][selected])) if selected.sum()>=16 else 0.
            preserve=float(np.clip((signal-2)/2,0,1));held=max([preserve]+list(s['preserve_history']))
            amount=float(np.clip((min(h,w)-20)/8,0,1))*(1-held)
            observations=[patch];positions=[(y,x)]
            for lag in [1,2,4]:
                if len(s['history'])<lag or len(self.bank)<lag:continue
                py,px,ph,pw=s['history'][-lag];oy=int(round(py+ph/2-h/2));ox=int(round(px+pw/2-w/2))
                if oy<0 or ox<0 or oy+h>height or ox+w>width:continue
                observations.append(self.bank[-lag][oy:oy+h,ox:ox+w]);positions.append((oy,ox))
            merged=np.median(np.stack(observations),axis=0) if len(observations)>=3 else patch
            total[y:y+h,x:x+w]+=merged*component;count[y:y+h,x:x+w]+=component
            for py,px in positions:
                mask[py:py+h,px:px+w]=np.maximum(mask[py:py+h,px:px+w],component)
                weight[py:py+h,px:px+w]=np.maximum(weight[py:py+h,px:px+w],amount*component)
            records.append(dict(y=y,x=x,h=h,w=w,missed=s['missed'],protection=held,weight=amount))
            s['history'].append(s['box']);s['preserve_history'].append(preserve)
            s['template']=cv2.GaussianBlur(patch,(0,0),.65);s['energy']=float(np.mean(np.abs(patch[component])))
        mask=cv2.dilate(mask,np.ones((7,7),np.uint8))*(1-protect)*self.road
        weight=cv2.dilate(weight,np.ones((7,7),np.uint8))*(1-protect)*self.road;weight[:,:64]=0
        merged=np.where(count>0,total/np.maximum(count,1),current)
        self.states=states;self.bank.append(current.copy())
        return mask.astype(np.float16).astype(np.float32),weight.astype(np.float16).astype(np.float32),merged,records

