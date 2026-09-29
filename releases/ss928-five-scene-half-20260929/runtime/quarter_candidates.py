"""Quarter-grid learned processing; all sixteen native pixel phases retained."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from extreme_candidates import load_extreme,BASE_CASE,prepare_inputs,FrontLayout

class QuarterFront(nn.Module):
    def __init__(self,kind='k4',width=16):
        super().__init__();self.kind=kind
        if kind=='k4':
            self.first=nn.Conv2d(9,32,4,stride=4);self.last=nn.Conv2d(32,16,1)
        else:
            self.first=nn.Conv2d(9,16,3,stride=2,padding=1);self.last=nn.Conv2d(16,16,3,stride=2,padding=1)
        with torch.no_grad():
            self.first.weight.zero_();self.first.bias.zero_();self.last.weight.zero_();self.last.bias.fill_(.5)
            if kind=='k4':
                for phase in range(16):
                    dy,dx=divmod(phase,4);self.first.weight[phase,8,dy,dx]=1;self.first.weight[phase+16,8,dy,dx]=-1;self.last.weight[phase,phase,0,0]=1;self.last.weight[phase,phase+16,0,0]=-1
            else:
                for phase in range(4):
                    dy,dx=divmod(phase,2);self.first.weight[phase,8,1+dy,1+dx]=1;self.first.weight[phase+4,8,1+dy,1+dx]=-1
                    for sign,index in ((1,8+phase),(-1,12+phase)):
                        self.first.weight[index,8,1+dy,1+dx]=sign;self.first.weight[index,:6,1+dy,1+dx]=-sign/6
                for phase in range(16):
                    dy,dx=divmod(phase,4);source=(dy%2)*2+dx%2;self.last.weight[phase,source,1+dy//2,1+dx//2]=1;self.last.weight[phase,source+4,1+dy//2,1+dx//2]=-1
        self.wide=width==32
        if self.wide:
            self.last=nn.Conv2d(32,32,1) if kind=='k4' else nn.Conv2d(16,32,3,stride=2,padding=1)
            with torch.no_grad():
                self.last.weight.zero_();self.last.bias.zero_()
                if kind=='k4':
                    for i in range(32):self.last.weight[i,i,0,0]=1
                else:
                    for phase in range(16):
                        dy,dx=divmod(phase,4);source=(dy%2)*2+dx%2
                        for sign,index in ((1,phase),(-1,phase+16)):
                            self.last.weight[index,source,1+dy//2,1+dx//2]=sign;self.last.weight[index,source+4,1+dy//2,1+dx//2]=-sign
    def forward(self,x):
        value=self.last(F.relu(self.first(x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last))))
        return F.relu(value) if self.wide else value

class QuarterOutput(nn.Module):
    def __init__(self,mode='pixel'):
        super().__init__();self.mode=mode;self.conv=nn.Conv2d(16,16,1)
        with torch.no_grad():
            self.conv.weight.zero_();self.conv.bias.fill_(-.5)
            for i in range(16):self.conv.weight[i,i,0,0]=1
        first=torch.zeros(16,4,2,2);second=torch.zeros(4,1,2,2)
        for y in range(4):
            for x in range(4):first[y*4+x,y%2*2+x%2,y//2,x//2]=1
        for y in range(2):
            for x in range(2):second[y*2+x,0,y,x]=1
        self.register_buffer('first',first);self.register_buffer('second',second);self.register_buffer('repeat3',torch.ones(1,1,3,3))
    def native(self,x):return F.pixel_shuffle(self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)),4)
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).clamp(0,1)*255
        if self.mode=='pixel':native=F.pixel_shuffle(phases,4)
        else:native=F.conv_transpose2d(F.conv_transpose2d(phases,self.first,stride=2),self.second,stride=2)
        return F.conv_transpose2d(native,self.repeat3,stride=3) if self.mode=='cascade' else F.interpolate(native,scale_factor=3,mode='nearest')

class QuarterSystem(nn.Module):
    def __init__(self,reference,depth=2,mode='pixel',front_kind='k4',width=16):
        super().__init__();self.input_half=True;self.input_channels=9;self.front=QuarterFront(front_kind,width);layers=[]
        for _ in range(depth):
            conv=nn.Conv2d(width,width,3,padding=1)
            with torch.no_grad():
                conv.weight.zero_();conv.bias.zero_()
                for i in range(width):conv.weight[i,i,1,1]=1
            layers.extend([conv,nn.ReLU()])
        self.body=nn.Sequential(*layers);self.tail=nn.Conv2d(width,16,3,padding=1)
        with torch.no_grad():
            self.tail.weight.zero_();self.tail.bias.zero_()
            for i in range(16):self.tail.weight[i,i,1,1]=1
        self.output=QuarterOutput(mode);self.reference=copy.deepcopy(reference)
        if width==32:
            with torch.no_grad():
                for i in range(16):self.tail.weight[i,i+16,1,1]=-1
                self.output.conv.bias.zero_()
            old=self.reference.project;self.reference.project=nn.Conv2d(old.in_channels,32,old.kernel_size,padding=old.padding)
            nn.init.zeros_(self.reference.project.weight);nn.init.zeros_(self.reference.project.bias)
    def forward(self,x,context):
        v=self.body(self.front(x));ref=self.reference.encode(context.to(dtype=self.front.first.weight.dtype,memory_format=torch.channels_last));ref=self.reference.project(ref+ref.mean((-2,-1),keepdim=True));ref=F.interpolate(ref,size=v.shape[-2:],mode='bilinear',align_corners=False);return self.output(self.tail(v+ref))

def load_quarter(scene,case,run_dir,device='cuda'):
    if not case.startswith('quarter'):return load_extreme(scene,case,run_dir,device=device)
    stem=case;layout=None
    for suffix in ('nhwc16','native5'):
        if stem.endswith('_'+suffix):layout=suffix;stem=stem[:-len(suffix)-1]
    quant=None
    for suffix in ('_w8allpo','_w8allpt'):
        if stem.endswith(suffix):quant=suffix;stem=stem[:-len(suffix)]
    depth=1 if 'body1' in stem else 2;mode='cascade' if 'cascade' in stem else 'deconv' if 'deconv' in stem else 'pixel';base=load_extreme(scene,BASE_CASE,run_dir,device=device);reference=base.core.model.global_reference
    kind='3x3' if '3x3' in stem else 'k4';width=32 if 'w32' in stem else 16;model=QuarterSystem(reference,depth,mode,kind,width);name=(f'{scene}_quarter_w32_{kind}_body{depth}_gtweak016000.pt' if 'gtweak' in stem else f'{scene}_quarter_w32_{kind}_body{depth}_qat008000.pt' if 'qat' in stem else f'{scene}_quarter_w32_{kind}_body{depth}_016000_stage1.pt' if 'stage1' in stem else f'{scene}_quarter_w32_{kind}_body{depth}_024000.pt') if width==32 else f'{scene}_quarter_{kind}_body{depth}_016000.pt';checkpoint=torch.load(Path(run_dir)/name,map_location='cpu',weights_only=True)
    if 'gtweak' in stem:model.tail=nn.Identity();model.output.conv=nn.Conv2d(32,16,3,padding=1)
    model.load_state_dict(checkpoint['model'],strict=True);model=model.to(device=device,dtype=torch.float16,memory_format=torch.channels_last).eval()
    if quant:
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,nn.Conv2d):
                    w=layer.weight.float();scale=(w.abs().amax((1,2,3),keepdim=True) if quant=='_w8allpo' else w.abs().max()).clamp_min(1e-9)/127;layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
    return FrontLayout(model,layout).eval() if layout else model


_base_load_quarter=load_quarter
def load_quarter(scene,case,run_dir,device='cuda'):
    if not case.startswith('cal_'):return _base_load_quarter(scene,case,run_dir,device=device)
    quant=None;stem=case[4:]
    for suffix in ('_w8allpo','_w8allpt'):
        if stem.endswith(suffix):quant=suffix;stem=stem[:-len(suffix)]
    model=_base_load_quarter(scene,stem,run_dir,device=device)
    checkpoint=torch.load(Path(run_dir)/f'{scene}_cal_{stem}.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model'],strict=True)
    if quant:
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    w=layer.weight.float();axes=(0,2,3) if isinstance(layer,nn.ConvTranspose2d) else (1,2,3);scale=(w.abs().amax(axes,keepdim=True) if quant=='_w8allpo' else w.abs().max()).clamp_min(1e-9)/127;layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
    return model.eval()
