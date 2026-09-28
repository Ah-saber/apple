"""Full 3x output layout alternatives avoiding Resize and fixed deconvolution."""
import torch
from torch import nn
from torch.nn import functional as F
from front_candidates import load_front_candidate,prepare_inputs

class RepeatOutput(nn.Module):
    def __init__(self,old,mode):
        super().__init__()
        self.mode=mode.removesuffix('4')
        if mode.endswith('4'):
            source=old.conv
            assert isinstance(source,nn.Conv2d) and source.out_channels>=4
            self.conv=nn.Conv2d(source.in_channels,4,source.kernel_size,
                                stride=source.stride,padding=source.padding,
                                dilation=source.dilation,groups=source.groups,
                                bias=source.bias is not None,
                                device=source.weight.device,dtype=source.weight.dtype)
            with torch.no_grad():
                self.conv.weight.copy_(source.weight[:4])
                if source.bias is not None:self.conv.bias.copy_(source.bias[:4])
        else:self.conv=old.conv
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))[:,:4].clamp(0,1)*255
        native=F.pixel_shuffle(phases,2)
        n,c,h,w=(int(v) for v in native.shape)
        shaped=native.reshape(n,c,h,1,w,1)
        if self.mode=='expand':
            return shaped.expand(n,c,h,3,w,3).reshape(n,c,h*3,w*3)
        if self.mode=='tile':
            return shaped.repeat(1,1,1,3,1,3).reshape(n,c,h*3,w*3)
        raise ValueError(self.mode)

def load_output_candidate(scene,name,run_dir,device='cuda'):
    if name=='rows32':return load_front_candidate(scene,name,run_dir,device=device)
    stem,mode=name.rsplit('__',1)
    if mode not in ('expand','tile','expand4','tile4'):raise ValueError(name)
    model=load_front_candidate(scene,stem,run_dir,device=device)
    model.output=RepeatOutput(model.output,mode).to(device=device).eval()
    return model.eval()
