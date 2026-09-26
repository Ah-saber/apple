"""Half-resolution trajectory with trimmed temporal means instead of medians."""
import torch
from torch.nn import functional as F

def trimmed(tensor):
    return (tensor.sum(dim=1,keepdim=True)-tensor.amin(dim=1,keepdim=True)-tensor.amax(dim=1,keepdim=True))/(tensor.shape[1]-2)

def trajectory_features_nozero(self,x):
    full_size=x.shape[-2:]
    x=F.avg_pool2d(x.float(),2)
    reference=trimmed(x[:,:6])
    residual=x-reference
    filters=[]
    for kh,kw in ((7,1),(1,7),(3,3)):
        filters.append(F.avg_pool2d(F.pad(residual,(kw//2,kw//2,kh//2,kh//2),mode='replicate'),(kh,kw),stride=1))
    filtered=torch.cat(filters,dim=0)
    center=trimmed(filtered)
    scale=trimmed((filtered-center).abs())*1.4826
    scale=F.avg_pool2d(F.pad(scale,(3,3,3,3),mode='replicate'),7,stride=1)
    floor=scale[...,::8,::8].mean(dim=(-2,-1),keepdim=True)*.6
    response=filtered[:,-3:]/torch.maximum(scale,floor.clamp_min(.001))
    z=response.reshape(3,x.shape[0],3,*x.shape[-2:])
    positive=z.amax(dim=0).clamp_min(0).clamp_max(10)
    negative=(-z).amax(dim=0).clamp_min(0).clamp_max(10)
    velocities=[(0,0)]+[(dy*step,dx*step) for step in (2,3,5,6)
               for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))]
    activity=torch.cat((positive,negative),dim=0)
    current,previous,older=activity[:,-1:],activity[:,-2:-1],activity[:,-3:-2]
    height,width=current.shape[-2:]
    previous_pad=F.pad(previous,(6,6,6,6))
    older_pad=F.pad(older,(12,12,12,12))
    max_pair=previous*older
    for dy,dx in velocities[1:]:
        prev=previous_pad[...,6-dy:6-dy+height,6-dx:6-dx+width]
        old=older_pad[...,12-2*dy:12-2*dy+height,12-2*dx:12-2*dx+width]
        max_pair=torch.maximum(max_pair,prev*old)
    support=torch.sqrt(max_pair+1e-8)
    supported=(current*support).clamp_max(20)
    pos_current,neg_current=current.chunk(2,dim=0)
    pos_supported,neg_supported=supported.chunk(2,dim=0)
    low=torch.cat((pos_current,pos_supported,neg_current,neg_supported),dim=1)
    return F.interpolate(low,size=full_size,mode='bilinear',align_corners=False)
