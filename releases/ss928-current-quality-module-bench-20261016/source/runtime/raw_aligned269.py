"""Refine box-center alignment from current RAW target texture, without future frames."""
from pathlib import Path
import cv2,numpy as np
from small_temporal258 import RawSmallTemporal258

CACHE269=Path('/tmp/rawir-small269-20261016')

class RawAligned269(RawSmallTemporal258):
    def align(self,old,xx,yy,dx,dy,current,owned,sigma):
        selected=(owned>.2)&(np.abs(current)>2*sigma)
        if selected.sum()<8:return super().align(old,xx,yy,dx,dy,current,owned,sigma)
        def candidate(sx,sy):
            warped=cv2.remap(old,xx+dx+sx,yy+dy+sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            valid=selected&(current*warped>0)&(np.abs(warped)>sigma)
            if valid.sum()<max(6,.3*selected.sum()):return np.inf,warped
            residual=warped-current
            for sign in [1,-1]:
                group=valid&(sign*current>0)
                if group.any():residual[group]-=np.median(residual[group])
            cost=float(np.minimum(np.abs(residual[valid])/(3*sigma),2).mean())+.15*(1-valid.sum()/selected.sum())+.01*(sx*sx+sy*sy)
            return cost,warped
        best_cost,best=candidate(0.,0.);best_shift=(0.,0.)
        for sy in [-2.,-1.,0.,1.,2.]:
            for sx in [-2.,-1.,0.,1.,2.]:
                cost,warped=candidate(sx,sy)
                if cost<best_cost:best_cost,best,best_shift=cost,warped,(sx,sy)
        cx,cy=best_shift
        for sy in [cy-.5,cy,cy+.5]:
            for sx in [cx-.5,cx,cx+.5]:
                cost,warped=candidate(sx,sy)
                if cost<best_cost:best_cost,best=cost,warped
        return best
