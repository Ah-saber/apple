"""Keep current RAW-owned foreground contrast during vacated-position cleanup."""
import json
from pathlib import Path
import cv2,numpy as np
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM

def project_tail(delta,compact,active,road,large_only=False,core_only=False):
    owner=((active if large_only else np.maximum(compact,active))>.15).astype(np.uint8)
    if core_only:
        # active195 inherits owner167's 7x7 dilation. Remove that margin.
        owner=cv2.erode(owner,np.ones((7,7),np.uint8))
        weight=np.clip(cv2.distanceTransform(owner,cv2.DIST_L2,3)/2,0,1)
    else:
        owner=cv2.dilate(owner,np.ones((3,3),np.uint8))
        distance=cv2.distanceTransform(1-owner,cv2.DIST_L2,3)
        weight=np.clip((3-distance)/3,0,1)
    background=(cv2.dilate(owner,np.ones((5,5),np.uint8))==0).astype(np.float32)
    count=cv2.blur(background,(63,63),borderType=cv2.BORDER_REFLECT)
    reference=cv2.blur(delta*background,(63,63),borderType=cv2.BORDER_REFLECT)/np.maximum(count,1e-6)
    reference=np.where(count>1e-4,reference,delta)
    return delta*(1-weight)+reference*road*weight

def prepare(number=289):
    from diagnose_surround210 import outputs
    from retained_output import retained_image
    from joint_context279 import CACHE282
    model_source=286 if number in [291,293] else 285
    for cap in ['capture_00','capture_01']:
        arrays=outputs(cap);shape=arrays['source'].shape
        compact=np.load(Path('/tmp/rawir-residual207-20261014')/'val_continuous'/cap/'mask207.npy',mmap_mode='r')
        active=np.load(Path('/tmp/rawir-residual195-20261014')/'val_continuous'/cap/'fields195.npy',mmap_mode='r')
        road=np.load(CACHE282/cap/'mask.npy',mmap_mode='r')
        out=Path('/tmp/rawir-residual%d-20261015'%number)/'val_continuous'/cap;out.mkdir(parents=True,exist_ok=True)
        head=np.lib.format.open_memmap(out/('head%d.npy'%number),mode='w+',dtype=np.float32,shape=shape)
        for i in range(len(head)):
            base=np.clip(np.clip(arrays['source'][i].astype(np.float32)+arrays['parent'][i],0,255)+arrays['current'][i],0,255)
            parent=retained_image(base,283,cap,i);proposal=retained_image(base,model_source,cap,i)
            delta=proposal[TOP:BOTTOM]-parent[TOP:BOTTOM]
            projected=project_tail(delta,compact[i,TOP:BOTTOM].astype(np.float32),active[i,2].astype(np.float32),road[i].astype(np.float32),large_only=number in [290,291,293],core_only=number==293)
            result=parent.copy();result[TOP:BOTTOM]=np.clip(parent[TOP:BOTTOM]+projected,0,255)
            head[i]=result-base
        head.flush();(out/'complete.json').write_text(json.dumps(dict(parent_complete=206,current_complete_parent=283,model_round=model_source,GT_used=False,feedback=False)))
        print('TAIL_PROJECTED',cap,flush=True)
    report=ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32';report.mkdir(parents=True,exist_ok=True)
    (report/'config.json').write_text(json.dumps(dict(parent=283,model_round=model_source,model_strength='selected original-train strength',foreground='active195' if number in [290,291,293] else 'current compact207 and active195',core_only=number==293,background_kernel=63),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=[289,290,291,293],default=289);args=p.parse_args()
    cv2.setNumThreads(2);prepare(args.round)
