"""Aggressive complete-model controls; SDK and target speed remain unverified."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from input_candidates import load_input,prepare_inputs
from body_candidates import BASE_CASE
BODY_RUN=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BODY-DISTILL-20260927')

class PlainFront(nn.Module):
    def __init__(self,kernel=3,last_kernel=1,width=16):
        super().__init__();assert width>=16
        self.first=nn.Conv2d(9,width,kernel,stride=2,padding=kernel//2)
        self.last=nn.Conv2d(width,4,last_kernel,padding=last_kernel//2)
        with torch.no_grad():
            self.first.weight.zero_();self.first.bias.zero_();self.last.weight.zero_();self.last.bias.zero_()
            center=kernel//2
            for phase in range(4):
                dy,dx=divmod(phase,2)
                self.first.weight[phase,8,center+dy,center+dx]=1
                self.first.weight[phase+4,8,center+dy,center+dx]=-1
                self.last.weight[phase,phase,last_kernel//2,last_kernel//2]=1
                self.last.weight[phase,phase+4,last_kernel//2,last_kernel//2]=-1
                # A small signed temporal difference initialized in the spare channels.
                for sign,index in ((1,8+phase),(-1,12+phase)):
                    self.first.weight[index,8,center+dy,center+dx]=sign
                    self.first.weight[index,:6,center+dy,center+dx]=-sign/6
    def forward(self,x):
        x=x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        return self.last(F.relu(self.first(x)))

class SplitOutput(nn.Module):
    def __init__(self,original):
        super().__init__();self.conv=copy.deepcopy(original.conv)
        kernel=torch.zeros((16,1,3,3),device=self.conv.weight.device,dtype=self.conv.weight.dtype)
        for y in range(3):
            for x in range(3):kernel[3*y+x,0,y,x]=1
        self.register_buffer('kernel',kernel)
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).clamp(0,1)*255
        value=F.conv_transpose2d(phases,self.kernel,stride=3)
        return F.interpolate(value,scale_factor=2,mode='nearest')

class NativeOutput(nn.Module):
    def __init__(self,original,mode='resize'):
        super().__init__();self.mode=mode;old=original.conv;dtype=old.weight.dtype;device=old.weight.device
        mix=torch.tensor([[2/3,1/3,0],[0,1/3,2/3]],device=device,dtype=torch.float64)
        transform=torch.einsum('ay,bx->abyx',mix,mix).reshape(4,9)
        self.conv=nn.Conv2d(16,4,3,padding=1,device=device,dtype=dtype)
        with torch.no_grad():
            self.conv.weight.copy_((transform@old.weight[:9].double().flatten(1)).reshape(4,16,3,3))
            self.conv.bias.copy_(transform@old.bias[:9].double())
        kernel=torch.zeros((4,1,2,2),device=device,dtype=dtype)
        for y in range(2):
            for x in range(2):kernel[y*2+x,0,y,x]=1
        self.register_buffer('kernel2',kernel)
        kernel6=torch.zeros((4,1,6,6),device=device,dtype=dtype)
        for y in range(2):
            for x in range(2):kernel6[y*2+x,0,3*y:3*y+3,3*x:3*x+3]=1
        self.register_buffer('kernel6',kernel6);self.register_buffer('repeat3',torch.ones((1,1,3,3),device=device,dtype=dtype))
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).clamp(0,1)*255
        if self.mode=='deconv':return F.conv_transpose2d(phases,self.kernel6,stride=6)
        native=F.conv_transpose2d(phases,self.kernel2,stride=2)
        return F.conv_transpose2d(native,self.repeat3,stride=3) if self.mode=='cascade' else F.interpolate(native,scale_factor=3,mode='nearest')

def load_extreme(scene,case,run_dir,device='cuda'):
    root=Path(run_dir);body_root=root/'body_models'
    if not body_root.is_dir():body_root=BODY_RUN
    if case in ('baseline',BASE_CASE,'body2_quantsearch','body3_quantsearch','combo_stats3_relu32_aligned16_half_output_fixed'):
        return load_input(scene,case,body_root,device=device)
    body='body2_quantsearch' if '_body2' in case else 'body3_quantsearch' if '_body3' in case else BASE_CASE
    model=load_input(scene,body,body_root,device=device)
    model.input_half=True
    if case.startswith('front'):
        tag=case.split('_',1)[0];k,last=map(int,tag[5:].split('x'));front=PlainFront(k,last)
        if '_trained' in case:
            ck=torch.load(root/f'{scene}_{tag}_008000.pt',map_location='cpu',weights_only=True)
            front.load_state_dict(ck['front'],strict=True)
        model.front=front.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
    if 'split3' in case:model.output=SplitOutput(model.output)
    if 'native4' in case:model.output=NativePixelOutput(model.output,aligned='alignpixel' in case) if 'pixel' in case else NativeOutput(model.output,'cascade' if 'cascade' in case else 'deconv' if 'deconv' in case else 'resize')
    if 'direct4' in case:
        output=DirectOutput4('cascade' if 'cascade' in case else 'resize')
        ck=torch.load(root/f'{scene}_direct4_008000.pt',map_location='cpu',weights_only=True);output.load_state_dict(ck['output'],strict=True)
        model.output=output.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
    return model.eval()

class FrontLayout(nn.Module):
    def __init__(self,model,layout):
        super().__init__();self.model=model;self.input_channels=16;self.input_half=True;self.input_layout=layout
        point=getattr(model.front,'temporal_conv',None)
        old=point if point is not None else model.front.first
        replacement=nn.Conv2d(16,old.out_channels,old.kernel_size,stride=old.stride,padding=old.padding,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():replacement.weight.zero_();replacement.weight[:,:9].copy_(old.weight);replacement.bias.copy_(old.bias)
        
        if point is not None:model.front.temporal_conv=replacement
        else:model.front.first=replacement
    def forward(self,x,context):
        if self.input_layout=='native5':x=x.reshape(x.shape[0],x.shape[2],x.shape[3],16)
        return self.model(x.permute(0,3,1,2),context)

_unpadded_load=load_extreme
def load_extreme(scene,case,run_dir,device='cuda'):
    for layout in ('nhwc16','native5'):
        if case.endswith('_'+layout):
            stem=case[:-len(layout)-1]
            if not stem.startswith('front'):raise ValueError('Aligned layout control requires direct plain front')
            return FrontLayout(_unpadded_load(scene,stem,run_dir,device=device),layout).eval()
    return _unpadded_load(scene,case,run_dir,device=device)


class DirectOutput4(nn.Module):
    def __init__(self,mode='resize'):
        super().__init__();self.mode=mode;self.conv=nn.ConvTranspose2d(16,1,4,stride=2,padding=1);self.register_buffer('repeat3',torch.ones(1,1,3,3))
    def native(self,x):return self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))
    def forward(self,x):
        gray=self.native(x).clamp(0,1)*255
        return F.conv_transpose2d(gray,self.repeat3,stride=3) if self.mode=='cascade' else F.interpolate(gray,scale_factor=3,mode='nearest')


class NativePixelOutput(NativeOutput):
    def __init__(self,original,aligned=False):
        super().__init__(original);self.aligned=aligned
        if aligned:
            old=self.conv;new=nn.Conv2d(16,16,3,padding=1,device=old.weight.device,dtype=old.weight.dtype)
            with torch.no_grad():new.weight.zero_();new.bias.zero_();new.weight[:4].copy_(old.weight);new.bias[:4].copy_(old.bias)
            self.conv=new
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))[:,:4].clamp(0,1)*255
        return F.interpolate(F.pixel_shuffle(phases,2),scale_factor=3,mode='nearest')

_layout_load=load_extreme
def load_extreme(scene,case,run_dir,device='cuda'):
    quant=None
    for suffix in ('_w8allpo','_w8allpt'):
        if case.endswith(suffix):quant=suffix;case=case[:-len(suffix)]
    if case.startswith('joint_direct4'):
        model=_layout_load(scene,'front3x1_trained_body2_direct4'+('_cascade' if 'cascade' in case else ''),run_dir,device=device)
        state=torch.load(Path(run_dir)/f'{scene}_joint_direct4_016000.pt',map_location='cpu',weights_only=True)
        model.front.load_state_dict(state['front'],strict=True);model.core.model.body.load_state_dict(state['body'],strict=True);model.output.load_state_dict(state['output'],strict=True)
    else:model=_layout_load(scene,case,run_dir,device=device)
    if quant:
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    w=layer.weight.float()
                    axes=(0,2,3) if isinstance(layer,nn.ConvTranspose2d) else (1,2,3)
                    scale=(w.abs().amax(axes,keepdim=True) if quant=='_w8allpo' else w.abs().max()).clamp_min(1e-9)/127
                    layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
        model.weight8_pressure_scope='all learned convolution weights; no bias/activation/SDK calibration simulation'
    return model.eval()

class PreservedFront(nn.Module):
    """Old statistics/front as pointwise temporal mixing plus two spatial convolutions.
    Mathematical equality precedes Half intermediate rounding and quantization.
    Current bypass uses signed positive/negative channels, valid for negative RAW.
    """
    def __init__(self,old,kind='mean',kernel=6):
        super().__init__();width=old.first.out_channels;out=((width+8+15)//16)*16;device=old.first.weight.device;dtype=old.first.weight.dtype;self.temporal_conv=None;self.kind=kind;self.kernel=kernel
        if hasattr(old,'temporal'):temporal=old.temporal.detach().double()[:,:,0,0]
        elif hasattr(old,'stats_weight'):temporal=torch.stack([old.stats_weight[4*i,:,0,0] for i in range(3)]).double()
        else:raise ValueError('Unknown preserved statistics front')
        if not torch.equal(temporal[0],torch.tensor([0.]*8+[1.],device=device,dtype=torch.float64)):raise ValueError('Current frame coefficient changed')
        if kind=='raw':channels=9;mix=temporal
        else:
            channels=16;self.temporal_conv=nn.Conv2d(9,16,1,device=device,dtype=dtype);gains=torch.ones(3,device=device,dtype=torch.float64)
            if kind=='sum':gains=temporal.abs().amax(1)
            with torch.no_grad():
                self.temporal_conv.weight.zero_();self.temporal_conv.bias.zero_()
                for i in range(3):self.temporal_conv.weight[i,:,0,0].copy_(temporal[i]/gains[i]);self.temporal_conv.weight[i+3,:,0,0].copy_(-temporal[i]/gains[i])
            mix=torch.zeros(3,16,device=device,dtype=torch.float64)
            for i in range(3):mix[i,i]=gains[i];mix[i,i+3]=-gains[i]
        z=torch.cat((64*(mix[0]-mix[1])[None].repeat(4,1),64*(mix[0]-mix[2])[None].repeat(4,1),mix[0:1].repeat(4,1)),0)
        shift=kernel-6;self.padding=2+shift;self.first=nn.Conv2d(channels,out,kernel,stride=2,padding=self.padding,device=device,dtype=dtype);self.last=nn.Conv2d(out,4,3,padding=1,device=device,dtype=dtype)
        folded=torch.zeros(width,channels,kernel,kernel,device=device,dtype=torch.float64);weight=old.first.weight.detach().double()
        for phase in range(4):
            dy,dx=divmod(phase,2)
            for ky in range(3):
                for kx in range(3):
                    for g in range(3):folded[:,:,2*ky+dy+shift,2*kx+dx+shift]+=weight[:,g*4+phase,ky,kx,None]*z[g*4+phase][None]
        with torch.no_grad():
            self.first.weight.zero_();self.first.bias.zero_();self.first.weight[:width].copy_(folded);self.first.bias[:width].copy_(old.first.bias);self.last.weight.zero_();self.last.bias.copy_(old.last.bias.double()*.025);self.last.weight[:,:width].copy_(old.last.weight.double()*.025)
            for phase in range(4):
                dy,dx=divmod(phase,2)
                self.first.weight[width+phase,:,2+dy+shift,2+dx+shift].copy_(mix[0]*64);self.first.weight[width+4+phase,:,2+dy+shift,2+dx+shift].copy_(-mix[0]*64);self.last.weight[phase,width+phase,1,1]=1/64;self.last.weight[phase,width+4+phase,1,1]=-1/64
    def forward(self,x):
        x=x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        if self.temporal_conv is not None:x=F.relu(self.temporal_conv(x))
        return self.last(F.relu(self.first(x)))

_pre_preserved_load=load_extreme
def load_extreme(scene,case,run_dir,device='cuda'):
    if not case.startswith('preserve_'):return _pre_preserved_load(scene,case,run_dir,device=device)
    kind_kernel=case.split('_')[1];kernel=int(kind_kernel[-1]);kind=kind_kernel[:-1];body='body3_quantsearch' if '_body3' in case else 'body2_quantsearch' if '_body2' in case else BASE_CASE
    model=_pre_preserved_load(scene,body,run_dir,device=device);model.input_half=True;model.front=PreservedFront(model.front,kind,kernel)
    if 'native4' in case:model.output=NativePixelOutput(model.output,aligned='alignpixel' in case)
    return model.eval()
