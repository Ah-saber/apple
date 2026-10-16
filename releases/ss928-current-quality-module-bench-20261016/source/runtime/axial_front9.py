"""Factor the costly night front spatial convolution, preserving both existing ReLUs."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class AxialSpatial(nn.Module):
    def __init__(self,source,rank):
        super().__init__();o,i,h,w=source.weight.shape
        if h!=w or source.stride!=(2,2) or source.groups!=1:raise ValueError('Expected square stride-two night front')
        matrix=source.weight.detach().double().permute(0,3,1,2).reshape(o*w,i*h)
        u,s,v=torch.linalg.svd(matrix,full_matrices=False)
        options=dict(device=source.weight.device,dtype=source.weight.dtype)
        self.vertical=nn.Conv2d(i,rank,(h,1),stride=(2,1),padding=(source.padding[0],0),bias=False,**options)
        self.horizontal=nn.Conv2d(rank,o,(1,w),stride=(1,2),padding=(0,source.padding[1]),bias=source.bias is not None,**options)
        with torch.no_grad():
            self.vertical.weight.copy_((s[:rank].sqrt()[:,None]*v[:rank]).reshape(rank,i,h,1))
            self.horizontal.weight.copy_((u[:,:rank]*s[:rank].sqrt()[None]).reshape(o,w,rank).permute(0,2,1)[:,:,None,:])
            if source.bias is not None:self.horizontal.bias.copy_(source.bias)
        self.energy=float(s[:rank].square().sum()/s.square().sum());self.rank=rank
    def forward(self,x):return self.horizontal(self.vertical(x))

class CurriculumFirst(nn.Module):
    def __init__(self,source,rank):
        super().__init__();self.original=copy.deepcopy(source).requires_grad_(False);self.compact=AxialSpatial(source,rank);self.alpha=1.
    @property
    def weight(self):return self.compact.vertical.weight
    def forward(self,x):
        value=self.compact(x)
        return value if self.alpha>=1 else self.original(x)*(1-self.alpha)+value*self.alpha

class RecentFront(nn.Module):
    def __init__(self,source,count):
        super().__init__();old=source.temporal_conv;self.input_count=count
        self.temporal_conv=nn.Conv2d(count,old.out_channels,1,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            self.temporal_conv.weight.copy_(old.weight[:,-count:])
            dropped=old.weight[:,:old.in_channels-count].sum(1,keepdim=True)
            self.temporal_conv.weight[:,:count-1].add_(dropped/(count-1))
            if old.bias is not None:self.temporal_conv.bias.copy_(old.bias)
        self.first=copy.deepcopy(source.first);self.last=copy.deepcopy(source.last)
    def forward(self,x):
        x=x[:,-self.input_count:].to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        return self.last(F.relu(self.first(F.relu(self.temporal_conv(x)))))

def rewrite_front(model,rank):
    old=model.board.source.front
    if model.kind=='special':model.board.source.front=RecentFront(old,5)
    front=model.board.source.front;source=front.first;front.first=CurriculumFirst(source,rank)
    # Constructor parameters are new tensors; freeze all before selecting the training scope.
    model.requires_grad_(False)
    o,i,h,w=source.weight.shape
    record=dict(rank=rank,energy=front.first.compact.energy,old_channels=old.temporal_conv.in_channels,
        inference_channels=front.temporal_conv.in_channels,old_first_coefficients=o*i*h*w,
        compact_first_coefficients=rank*i*h+o*rank*w,
        old_first_MAC=o*i*h*w*512*640,
        compact_first_MAC=rank*i*h*512*1280+o*rank*w*512*640,
        existing_temporal_and_spatial_relu_preserved=True,activation_between_axial_layers=False,
        initialization_equivalence='Rank truncation; special nine-to-five temporal DC preserved, not exact on changing frames')
    return record

if __name__=='__main__':
    import json
    from target_residual import current_setup
    from causal_detail import reference_board
    records=[]
    for scene in ['night_ordinary','night_special']:
        kind,config,df,box,count,base,_,profile=current_setup(scene)
        row=next(r for r in df(config,'train').records if r['scene_id']==scene)
        for rank in [6,8]:
            model=reference_board(base,count,kind,box(row),scene);records.append(dict(scene=scene,**rewrite_front(model,rank)))
    print(json.dumps(records,indent=2))
