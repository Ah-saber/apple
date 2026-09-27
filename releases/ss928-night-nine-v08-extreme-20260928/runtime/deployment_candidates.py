"""Reference layout controls and projection-before-spatial-mean equivalence."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from quarter_candidates import load_quarter,prepare_inputs
from extreme_candidates import FrontLayout

class PaddedReference(nn.Module):
    def __init__(self,old):
        super().__init__()
        def conv(layer):
            if layer.groups!=1 or layer.in_channels not in (1,12) or layer.out_channels not in (12,16,32):raise ValueError('Unsupported reference geometry')
            ni=16 if layer.in_channels==12 else layer.in_channels;no=16 if layer.out_channels==12 else layer.out_channels
            new=nn.Conv2d(ni,no,layer.kernel_size,stride=layer.stride,padding=layer.padding,dilation=layer.dilation,device=layer.weight.device,dtype=layer.weight.dtype)
            with torch.no_grad():new.weight.zero_();new.bias.zero_();new.weight[:layer.out_channels,:layer.in_channels].copy_(layer.weight);new.bias[:layer.out_channels].copy_(layer.bias)
            return new
        def sequence(layers):return nn.Sequential(*(conv(layer) if isinstance(layer,nn.Conv2d) else copy.deepcopy(layer) for layer in layers))
        self.encoder=sequence(old.encoder);self.pyramid=nn.ModuleList([sequence(branch) for branch in old.pyramid]) if hasattr(old,'pyramid') else nn.ModuleList();self.project=conv(old.project)
    def encode(self,x):
        value=self.encoder(x)
        for branch in self.pyramid:value=value+F.interpolate(branch(x),size=(64,64),mode='bilinear',align_corners=False)
        return value

class ReferenceMeanFirst(nn.Module):
    def __init__(self,old):
        super().__init__();self.reference=copy.deepcopy(old);self.project=nn.Identity()
        with torch.no_grad():self.reference.project.bias.mul_(.5)
    def encode(self,x):return self.reference.project(self.reference.encode(x))

def load_deployment(scene,case,run_dir,device='cuda'):
    if case.startswith('diag_act'):
        _,mode,stem=case.split('_',2)
        from activation_candidates import attach_pressure
        return attach_pressure(load_deployment(scene,stem,run_dir,device=device),scene,stem,run_dir,mode)
    if case.startswith('cal_graph_'):
        stem=case[10:];quant=None
        for suffix in ('sdk_w8po','sdk_w8pt'):
            if stem.endswith('_'+suffix):quant=suffix;stem=stem[:-len(suffix)-1]
        model=load_deployment(scene,stem,run_dir,device=device)
        from pathlib import Path
        ck=torch.load(Path(run_dir)/f'{scene}_cal_{stem}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'],strict=True)
        if quant:
            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer,nn.Conv2d):
                        w=layer.weight.float();scale=(w.abs().amax((1,2,3),keepdim=True) if quant=='sdk_w8po' else w.abs().max()).clamp_min(1e-9)/127;layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
        return model.eval()
    stem=case;meanfirst=False;pad=False;interface=None;block=None;mapping=None;fused=False;quant=None;layout=None;gelu=None;dither=None;refsplit=False;kernel7=False
    for suffix in ("sdk_w8po","sdk_w8pt"):
        if stem.endswith("_"+suffix):quant=suffix;stem=stem[:-len(suffix)-1]
    for suffix in ('f32io','f32in','f32out'):
        if stem.endswith('_'+suffix):interface=suffix;stem=stem[:-len(suffix)-1]
    for suffix in ('block16_split_full','block16_full','block16'):
        if stem.endswith('_'+suffix):block=suffix;stem=stem[:-len(suffix)-1]
    for suffix in ('outdeconv','outcascade','outpixel'):
        if stem.endswith('_'+suffix):mapping=suffix;stem=stem[:-len(suffix)-1]
    if stem.endswith('_refsplit'):refsplit=True;stem=stem[:-9]
    if stem.endswith('_kernel7'):kernel7=True;stem=stem[:-8]
    for suffix in ('ditherphase','ditherfloat'):
        if stem.endswith('_'+suffix):dither=suffix;stem=stem[:-len(suffix)-1]
    if stem.endswith('_fusedtail'):fused=True;stem=stem[:-10]
    for suffix,count in (('gelu13',13),('gelu25',25)):
        if stem.endswith('_'+suffix):gelu=count;stem=stem[:-len(suffix)-1]
    for suffix in ('layout16','layout5'):
        if stem.endswith('_'+suffix):layout='nhwc16' if suffix=='layout16' else 'native5';stem=stem[:-len(suffix)-1]
    if stem.endswith('_meanfirst'):meanfirst=True;stem=stem[:-10]
    if stem.endswith('_refpad16'):pad=True;stem=stem[:-9]
    model=load_quarter(scene,stem,run_dir,device=device)
    if layout:model=FrontLayout(model,layout).eval()
    parent=getattr(model,'model',model);owner=parent.core.model if hasattr(parent,'core') else parent
    if kernel7:
        layer=parent.front.first
        if layer.kernel_size!=(6,6) or layer.stride!=(2,2) or layer.padding!=(2,2):raise ValueError('Kernel7 control expects preserved6 front')
        replacement=nn.Conv2d(layer.in_channels,layer.out_channels,7,stride=2,padding=3,device=layer.weight.device,dtype=layer.weight.dtype)
        with torch.no_grad():replacement.weight.zero_();replacement.weight[:,:,1:,1:].copy_(layer.weight);replacement.bias.copy_(layer.bias)
        parent.front.first=replacement
    if pad or meanfirst:
        name='global_reference' if hasattr(owner,'global_reference') else 'reference';reference=getattr(owner,name)
        if pad:reference=PaddedReference(reference)
        if meanfirst:reference=ReferenceMeanFirst(reference)
        setattr(owner,name,reference)
    if gelu:replace_reference_gelu(getattr(owner,'global_reference' if hasattr(owner,'global_reference') else 'reference'),gelu)
    if fused:
        if hasattr(parent,'core') or not isinstance(owner.tail,nn.Conv2d) or parent.output.conv.kernel_size!=(1,1):raise ValueError('Exact tail fusion requires quarter output1 projection')
        tail,projection=owner.tail,parent.output.conv
        layer=nn.Conv2d(tail.in_channels,projection.out_channels,3,padding=1,device=tail.weight.device,dtype=tail.weight.dtype)
        with torch.no_grad():
            matrix=projection.weight.double()[:,:,0,0];layer.weight.copy_(torch.einsum('oi,ickl->ockl',matrix,tail.weight.double()));layer.bias.copy_(matrix@tail.bias.double()+projection.bias.double())
        owner.tail=nn.Identity();parent.output.conv=layer
    if refsplit:
        replacement=SplitReferenceQuarter(parent)
        if hasattr(model,'model'):model.model=replacement
        else:model=replacement
        parent=replacement;owner=parent
    if dither:parent.output=DitherPhaseOutput(parent.output,2 if hasattr(parent,'core') else 4,dither=='ditherfloat')
    if mapping:
        parent.output=MappedOutput(parent.output,2 if hasattr(parent,'core') else 4,mapping[3:])
    if block:
        factor=2 if hasattr(parent,'core') else 4
        parent.output=BlockedOutput(parent.output,factor,full=block!='block16',split='split' in block)
        model.diagnostic_blocked_output=block=='block16'
    if quant:
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    w=layer.weight.float();axes=(0,2,3) if isinstance(layer,nn.ConvTranspose2d) else (1,2,3);scale=(w.abs().amax(axes,keepdim=True) if quant=='sdk_w8po' else w.abs().max()).clamp_min(1e-9)/127;layer.weight.copy_((w/scale).round().clamp(-127,127)*scale)
    model=model.eval()
    return DeclaredInterface(model,interface).eval() if interface else model


class DeclaredInterface(nn.Module):
    def __init__(self,model,mode):
        super().__init__();self.model=model;self.output_float=mode in ('f32io','f32out');self.input_half=mode not in ('f32io','f32in');self.input_channels=getattr(model,'input_channels',9);self.input_layout=getattr(model,'input_layout','nchw')
    def forward(self,x,context):
        value=self.model(x,context)
        return value.float() if self.output_float else value

class BlockedOutput(nn.Module):
    """Generate row16 blocks directly from low-resolution phases.
    Full mode restores gray layout INSIDE the graph; blocked mode is diagnostic.
    """
    def __init__(self,old,factor=2,full=True,split=False):
        super().__init__();self.conv=old.conv;self.factor=factor;self.full=full;self.split=split;self.group=16//factor;scale=factor*3
        channels=self.group*factor*factor;device=self.conv.weight.device;dtype=self.conv.weight.dtype
        kernel=torch.zeros(channels,16,3,scale,device=device,dtype=dtype)
        horizontal=torch.zeros(channels,channels,1,scale,device=device,dtype=dtype);vertical=torch.zeros(channels,16,3,1,device=device,dtype=dtype)
        for fy in range(self.group):
            for py in range(factor):
                for px in range(factor):
                    channel=fy*factor*factor+py*factor+px
                    for sy in range(3):
                        y=fy*scale+py*3+sy;destination=y%16;ky=y//16;vertical[channel,destination,ky,0]=1
                        for sx in range(3):kernel[channel,destination,ky,px*3+sx]=1
                    for sx in range(3):horizontal[channel,channel,0,px*3+sx]=1
        self.register_buffer('kernel',kernel);self.register_buffer('horizontal',horizontal);self.register_buffer('vertical',vertical)
    def forward(self,x):
        p=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))[:,:self.factor**2].clamp(0,1)*255
        n,c,h,w=p.shape
        if h%self.group:raise ValueError('Feature height must be divisible by row grouping')
        packed=p.reshape(n,c,h//self.group,self.group,w).permute(0,3,1,2,4).reshape(n,c*self.group,h//self.group,w)
        if self.split:
            blocked=F.conv_transpose2d(F.conv_transpose2d(packed,self.horizontal,stride=(1,self.factor*3)),self.vertical,stride=(3,1))
        else:blocked=F.conv_transpose2d(packed,self.kernel,stride=(3,self.factor*3))
        if self.full:return blocked.permute(0,2,1,3).reshape(n,1,blocked.shape[2]*16,blocked.shape[3])
        return blocked


class MappedOutput(nn.Module):
    """Preserve the learned projection and compare full-output mappings."""
    def __init__(self,old,factor,mode):
        super().__init__();self.conv=old.conv;self.factor=factor;self.mode=mode;device=self.conv.weight.device;dtype=self.conv.weight.dtype;c=self.conv.out_channels
        native=torch.zeros(c,1,factor,factor,device=device,dtype=dtype);full=torch.zeros(c,1,factor*3,factor*3,device=device,dtype=dtype)
        for py in range(factor):
            for px in range(factor):
                native[py*factor+px,0,py,px]=1;full[py*factor+px,0,py*3:py*3+3,px*3:px*3+3]=1
        self.register_buffer('native_kernel',native);self.register_buffer('full_kernel',full);self.register_buffer('repeat3',torch.ones(1,1,3,3,device=device,dtype=dtype))
    def forward(self,x):
        p=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).clamp(0,1)*255
        if self.mode=='deconv':return F.conv_transpose2d(p,self.full_kernel,stride=self.factor*3)
        if self.mode=='pixel':native=F.pixel_shuffle(p[:,:self.factor**2],self.factor)
        else:native=F.conv_transpose2d(p,self.native_kernel,stride=self.factor)
        return F.conv_transpose2d(native,self.repeat3,stride=3) if self.mode=='cascade' else F.interpolate(native,scale_factor=3,mode='nearest')

class PiecewiseGelu(nn.Module):
    """Fixed linear segments approximating GELU; bound checked independently.
    Dense Conv1/ReLU/Conv1 keep the small reference on common NPU operators.
    """
    def __init__(self,channels,count,device,dtype):
        super().__init__();knots=torch.linspace(-3,3,count,dtype=torch.float64,device=device);values=F.gelu(knots);values[0]=0;values[-1]=3;slopes=(values[1:]-values[:-1])/(knots[1:]-knots[:-1]);increments=torch.cat((slopes[:1],slopes[1:]-slopes[:-1],1-slopes[-1:]));padded=((channels*count+15)//16)*16
        self.first=nn.Conv2d(channels,padded,1,device=device,dtype=dtype);self.last=nn.Conv2d(padded,channels,1,device=device,dtype=dtype)
        with torch.no_grad():
            self.first.weight.zero_();self.first.bias.zero_();self.last.weight.zero_();self.last.bias.zero_()
            for c in range(channels):
                for i in range(count):self.first.weight[c*count+i,c,0,0]=1;self.first.bias[c*count+i]=-knots[i];self.last.weight[c,c*count+i,0,0]=increments[i]
        self.requires_grad_(False)
    def forward(self,x):return self.last(F.relu(self.first(x)))

def replace_reference_gelu(reference,count):
    replaced=0
    for module in list(reference.modules()):
        if isinstance(module,nn.Sequential):
            previous=None
            for i,layer in enumerate(list(module.children())):
                if isinstance(layer,nn.Conv2d):previous=layer
                elif isinstance(layer,nn.GELU):
                    if previous is None:raise ValueError('GELU width not known')
                    module[i]=PiecewiseGelu(previous.out_channels,count,previous.weight.device,previous.weight.dtype);replaced+=1
    if replaced==0:raise ValueError('No reference GELU found')
    return replaced

class DitherPhaseOutput(nn.Module):
    """Static sub-gray offsets preserve fractional RAW means after byte display.
    Offsets are fixed within each3x3 display cell, not learned3x detail.
    """
    def __init__(self,old,factor=4,display_float=False):
        super().__init__();self.conv=old.conv;self.factor=factor;scale=factor*3;count=scale**2;padded=((count+15)//16)*16;dtype=torch.float32 if display_float else self.conv.weight.dtype;self.fixed=nn.Conv2d(self.conv.out_channels,padded,1,device=self.conv.weight.device,dtype=dtype)
        with torch.no_grad():
            self.fixed.weight.zero_();self.fixed.bias.zero_()
            for py in range(factor):
                for px in range(factor):
                    for sy in range(3):
                        for sx in range(3):
                            index=(py*3+sy)*scale+px*3+sx;self.fixed.weight[index,py*factor+px,0,0]=1;self.fixed.bias[index]=(sy*3+sx-4)/9
        self.fixed.requires_grad_(False)
    def forward(self,x):
        p=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).clamp(0,1)*255;p=p.to(dtype=self.fixed.weight.dtype);value=self.fixed(p)[:,: (self.factor*3)**2].clamp(0,255);return F.pixel_shuffle(value,self.factor*3)

class SplitReferenceQuarter(nn.Module):
    """Fold reference projection through the linear output convolution.
    Reference interpolation shrinks32 to16 channels; constant channel preserves
    every3x3 border contribution of the old reference bias.
    """
    def __init__(self,old):
        super().__init__()
        if not isinstance(old.tail,nn.Identity) or old.output.conv.kernel_size!=(3,3) or old.output.conv.in_channels!=32:raise ValueError('Reference split needs fused quarter projection32->16 k3')
        project=old.reference.project
        if not isinstance(project,nn.Conv2d) or project.kernel_size!=(1,1):raise ValueError('Reference split needs unmodified reference project1')
        if project.in_channels not in (12,16):raise ValueError('Reference embedding width unsupported')
        if project.in_channels==16 and torch.count_nonzero(project.weight[:,12:]):raise ValueError('Expected only zero padded reference channels')
        self.front=old.front;self.body=old.body;self.tail=old.tail;self.output=old.output;self.reference=copy.deepcopy(old.reference);self.reference.project=nn.Identity();self.input_half=True;self.input_channels=getattr(old,'input_channels',9);self.refsplit=True;device=project.weight.device;dtype=project.weight.dtype
        self.reference_embed=nn.Conv2d(project.in_channels,16,1,device=device,dtype=dtype);self.reference_phase=nn.Conv2d(16,16,3,padding=1,bias=False,device=device,dtype=dtype)
        with torch.no_grad():
            self.reference_embed.weight.zero_();self.reference_embed.bias.zero_()
            for i in range(12):self.reference_embed.weight[i,i,0,0]=1
            self.reference_embed.bias[12]=1;self.reference_phase.weight.zero_();out=self.output.conv.weight.double();matrix=project.weight.double()[:,:,0,0];bias=project.bias.double();self.reference_phase.weight[:,:12].copy_(torch.einsum('oikl,ic->ockl',out,matrix[:,:12]));self.reference_phase.weight[:,12].copy_(torch.einsum('oikl,i->okl',out,bias))
    def forward(self,x,context):
        v=self.tail(self.body(self.front(x)));local=self.output.conv(v);r=self.reference.encode(context.to(dtype=self.front.first.weight.dtype,memory_format=torch.channels_last));r=self.reference_embed(r+r.mean((-2,-1),keepdim=True));r=F.interpolate(r,size=v.shape[-2:],mode='bilinear',align_corners=False);phases=local+self.reference_phase(r);return finish_quarter_phases(self.output,phases)

def finish_quarter_phases(output,p):
    """Existing complete output mappings, without a second learned projection."""
    if isinstance(output,DitherPhaseOutput):
        v=output.fixed((p.clamp(0,1)*255).to(output.fixed.weight.dtype))[:,:144].clamp(0,255);return F.pixel_shuffle(v,12)
    p=p.clamp(0,1)*255
    if isinstance(output,BlockedOutput):
        n,c,h,w=p.shape;p=p.reshape(n,c,h//output.group,output.group,w).permute(0,3,1,2,4).reshape(n,c*output.group,h//output.group,w)
        b=F.conv_transpose2d(F.conv_transpose2d(p,output.horizontal,stride=(1,12)),output.vertical,stride=(3,1)) if output.split else F.conv_transpose2d(p,output.kernel,stride=(3,12))
        return b.permute(0,2,1,3).reshape(n,1,b.shape[2]*16,b.shape[3]) if output.full else b
    if isinstance(output,MappedOutput):
        if output.mode=='deconv':return F.conv_transpose2d(p,output.full_kernel,stride=12)
        native=F.pixel_shuffle(p,4) if output.mode=='pixel' else F.conv_transpose2d(p,output.native_kernel,stride=4)
        return F.conv_transpose2d(native,output.repeat3,stride=3) if output.mode=='cascade' else F.interpolate(native,scale_factor=3,mode='nearest')
    if output.mode=='pixel':native=F.pixel_shuffle(p,4)
    else:native=F.conv_transpose2d(F.conv_transpose2d(p,output.first,stride=2),output.second,stride=2)
    return F.conv_transpose2d(native,output.repeat3,stride=3) if output.mode=='cascade' else F.interpolate(native,scale_factor=3,mode='nearest')
