"""Continue model-side optimization without altering nine-frame inputs."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from structural_candidates import load_structural,ReducedPhaseOutput,SpaceStatisticsFront

class PaddedInputFront(nn.Module):
    """Sixteen input channels: nine real frames and seven zero channels."""
    def __init__(self,original,channels=16,kernel_size=2):
        super().__init__();w=original.stats_weight
        self.padding=1 if kernel_size==3 else 0
        kernel=torch.zeros((12,channels,kernel_size,kernel_size),device=w.device,dtype=w.dtype);kernel[:,:9,self.padding:,self.padding:]=w
        self.register_buffer('weight',kernel);self.first=copy.deepcopy(original.first);self.last=copy.deepcopy(original.last)
    def forward(self,x):
        if self.padding and (x.shape[-2]%2 or x.shape[-1]%2):raise ValueError('Statistics3 requires even sensor dimensions')
        if self.padding:x=x.to(dtype=self.weight.dtype,memory_format=torch.channels_last)
        stats=F.conv2d(x,self.weight,stride=2,padding=self.padding).float()
        current,old,recent=stats[:,:4],stats[:,4:8],stats[:,8:]
        v=torch.cat(((current-old)*64,(current-recent)*64,current),1).to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        return current+self.last(F.relu(self.first(v))).float()*.025

def prepare_inputs(model,x,context):
    """Prepare the declared input representation outside the inference interval.
    No temporal statistics, reference processing or output expansion is moved here.
    """
    channels=getattr(model,'input_channels',9)
    if getattr(model,'input_half',False):x=x.half()
    if channels==16:x=torch.cat((x,torch.zeros_like(x[:,:7])),1)
    layout=getattr(model,'input_layout','nchw')
    if layout in ('nhwc16','native5'):
        x=x.permute(0,2,3,1).contiguous()
        if layout=='native5':x=x.unsqueeze(1)
    return x,context

class RectifiedStatisticsFront(nn.Module):
    """Represent signed statistics by positive/negative ReLU pairs.
    Supports signed normalized RAW; no assumption that all inputs are positive.
    """
    def __init__(self,original,aligned=False,kernel_size=2):
        super().__init__();w=original.stats_weight;self.negative_start=16 if aligned else 12
        if aligned:
            weight=torch.zeros((32,9,2,2),device=w.device,dtype=w.dtype);weight[:12]=w;weight[16:28]=-w
        else:weight=torch.cat((w,-w),0)
        self.padding=1 if kernel_size==3 else 0
        if kernel_size==3:
            padded=torch.zeros((*weight.shape[:2],3,3),device=w.device,dtype=w.dtype);padded[:,:,1:,1:]=weight;weight=padded
        self.register_buffer('weight',weight)
        self.register_buffer('bias',torch.zeros(weight.shape[0],device=w.device,dtype=w.dtype))
        self.first=copy.deepcopy(original.first);self.last=copy.deepcopy(original.last)
    def forward(self,x):
        if self.padding and (x.shape[-2]%2 or x.shape[-1]%2):raise ValueError('Statistics3 requires even sensor dimensions')
        x=x.to(dtype=self.weight.dtype,memory_format=torch.channels_last)
        v=F.relu(F.conv2d(x,self.weight,self.bias,stride=2,padding=self.padding))
        stats=v[:,:12].float()-v[:,self.negative_start:self.negative_start+12].float();current,old,recent=stats[:,:4],stats[:,4:8],stats[:,8:]
        z=torch.cat(((current-old)*64,(current-recent)*64,current),1).to(dtype=self.first.weight.dtype,memory_format=torch.channels_last)
        return current+self.last(F.relu(self.first(z))).float()*.025

class NativeFourOutput(nn.Module):
    def __init__(self,original,kind='linear'):
        super().__init__();self.kind=kind;dtype=original.weight.dtype;device=original.weight.device
        w=original.weight.detach().float().reshape(2,3,2,3,16,3,3).mean((1,3)).reshape(4,16,3,3)
        b=original.bias.detach().float().reshape(2,3,2,3).mean((1,3)).reshape(4)
        self.first=nn.Conv2d(16,4 if kind=='linear' else 8,3,padding=1,device=device,dtype=dtype)
        with torch.no_grad():
            if kind=='linear':self.first.weight.copy_(w);self.first.bias.copy_(b)
            else:
                self.first.weight.copy_(w.repeat(2,1,1,1));self.first.bias.copy_(torch.cat((b,b-1)))
        self.last=None
        if kind!='linear':
            self.last=nn.Conv2d(8,4,1,device=device,dtype=dtype)
            with torch.no_grad():
                self.last.weight.zero_();self.last.bias.zero_()
                for i in range(4):self.last.weight[i,i,0,0]=1;self.last.weight[i,i+4,0,0]=-1
    def phases(self,x):
        y=self.first(x.to(dtype=self.first.weight.dtype,memory_format=torch.channels_last))
        return self.last(F.relu(y)) if self.last is not None else y
    def forward(self,x):return F.interpolate(F.pixel_shuffle(self.phases(x),2),scale_factor=3,mode='nearest')

class DitherNativeOutput(nn.Module):
    """Encode native fractional gray means with fixed sub-gray phase offsets.
    This adds a deterministic 3x3 pattern; it does not restore 3x texture.
    """
    def __init__(self,base,mode='fixed'):
        super().__init__();self.base=base;self.mode=mode;w=base.first.weight
        tile=(torch.arange(9,device=w.device,dtype=torch.float32).reshape(3,3)-4)/9/255
        self.register_buffer('tile',tile[None,None])
        kernel=torch.zeros((5,1,6,6),device=w.device,dtype=w.dtype)
        for y in range(2):
            for x in range(2):kernel[y*2+x,0,y*3:y*3+3,x*3:x*3+3]=1
        kernel[4,0]=(tile.repeat(2,2)*255).to(w.dtype);self.register_buffer('kernel',kernel)
    def forward(self,x):
        phases=self.base.phases(x)
        if self.mode=='fixed':
            z=torch.cat((phases,torch.full_like(phases[:,:1],1/255)),1)
            return F.conv_transpose2d(z,self.kernel,stride=6)
        native=F.pixel_shuffle(phases,2);h,w=native.shape[-2:]
        return F.interpolate(native.float(),scale_factor=3,mode='nearest')+self.tile.repeat(1,1,h,w)

class FusedNineOutput(nn.Module):
    def __init__(self,projection,factor=3,bias_mode='channel'):
        super().__init__();self.factor=factor;self.bias_mode=bias_mode;rest=6//factor
        if factor not in (3,6):raise ValueError('Expected stride3 or stride6')
        old=projection.weight.detach();bias=projection.bias.detach();kernel=torch.zeros((17 if bias_mode=='channel' else 16,1,3*factor,3*factor),device=old.device,dtype=old.dtype)
        for py in range(factor):
            for px in range(factor):
                index=(py if factor==3 else py//2)*3+(px if factor==3 else px//2)
                for ky in range(3):
                    for kx in range(3):kernel[:16,0,(2-ky)*factor+py,(2-kx)*factor+px]=old[index,:,ky,kx]
                if bias_mode=='channel':kernel[16,0,factor+py,factor+px]=bias[index]
        self.register_buffer('weight',kernel.contiguous(memory_format=torch.channels_last))
        tile=bias.reshape(1,1,3,3)
        if factor==6:tile=tile.repeat_interleave(2,2).repeat_interleave(2,3)
        self.register_buffer('bias_tile',tile)
    def forward(self,x):
        x=x.to(dtype=self.weight.dtype,memory_format=torch.channels_last);h,w=x.shape[-2:]
        if self.bias_mode=='channel':x=torch.cat((x,torch.ones((x.shape[0],1,h,w),device=x.device,dtype=x.dtype)),1).contiguous(memory_format=torch.channels_last)
        y=F.conv_transpose2d(x,self.weight,stride=self.factor,padding=self.factor)
        if self.bias_mode=='image':y=y+self.bias_tile.repeat(1,1,h,w)
        return F.interpolate(y,scale_factor=2,mode='nearest') if self.factor==3 else y

class AlignedPhaseOutput(nn.Module):
    """Pad learned nine phases to16 logical channels, preserving nine coefficients."""
    def __init__(self,projection,mode='nearest'):
        super().__init__();self.mode=mode;w=projection.weight
        self.conv=nn.Conv2d(16,16,3,padding=1,device=w.device,dtype=w.dtype).to(memory_format=torch.channels_last)
        with torch.no_grad():
            self.conv.weight.zero_();self.conv.bias.zero_();self.conv.weight[:9].copy_(w);self.conv.bias[:9].copy_(projection.bias)
            if mode=='folded_fixed':self.conv.weight.mul_(255);self.conv.bias.mul_(255)
        kernel=torch.zeros((16,1,6,6),device=w.device,dtype=w.dtype)
        for y in range(3):
            for x in range(3):kernel[y*3+x,0,y*2:y*2+2,x*2:x*2+2]=1
        self.register_buffer('kernel',kernel)
    def forward(self,x):
        y=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))
        if self.mode in ('nearest','nearest_half_output'):
            y=y[:,:9]
            if self.mode=='nearest':y=y.float()
            y=y.clamp(0,1)*255
            return F.interpolate(F.pixel_shuffle(y,3),scale_factor=2,mode='nearest')
        if self.mode=='half_output_fixed':return F.conv_transpose2d(y.clamp(0,1)*255,self.kernel,stride=6)
        y=y.clamp(0,255 if self.mode=='folded_fixed' else 1)
        y=F.conv_transpose2d(y,self.kernel,stride=6).float()
        return y if self.mode=='folded_fixed' else y*255

class TailFusedOutput(nn.Module):
    """Compose linear Tail3 and projection3 into Conv5; optional true edge bands."""
    def __init__(self,tail,projection,edges=True):
        super().__init__();self.edges=edges;self.tail=copy.deepcopy(tail);self.projection=copy.deepcopy(projection)
        linear=tail
        if not isinstance(linear,nn.Conv2d):
            import ast,inspect,textwrap
            if not isinstance(tail,nn.Sequential) or len(tail)!=2 or not isinstance(tail[0],nn.Identity):raise ValueError('Unsupported Tail wrapper')
            block=tail[1];tree=ast.parse(textwrap.dedent(inspect.getsource(type(block).forward)));body=tree.body[0].body
            if len(body)!=1 or not isinstance(body[0],ast.Return) or not isinstance(body[0].value,ast.Call):raise ValueError('Tail must be one linear call')
            call=body[0].value
            if not isinstance(call.func,ast.Attribute) or call.func.attr!='rep_conv' or not isinstance(call.func.value,ast.Name) or call.func.value.id!='self' or len(call.args)!=1 or not isinstance(call.args[0],ast.Name) or call.args[0].id!='x' or call.keywords:raise ValueError('Tail forward differs from frozen linear wrapper')
            linear=block.rep_conv
        if not isinstance(linear,nn.Conv2d) or linear.weight.shape!=(16,16,3,3) or linear.stride!=(1,1) or linear.padding!=(1,1) or linear.groups!=1 or linear.dilation!=(1,1):raise ValueError('Require linear Tail16 k3 p1')
        wt=linear.weight.detach().cpu().double();wo=projection.weight.detach().cpu().double();bt=linear.bias.detach().cpu().double();bo=projection.bias.detach().cpu().double()
        w=torch.zeros((9,16,5,5),dtype=torch.float64)
        for oy in range(3):
            for ox in range(3):
                for ty in range(3):
                    for tx in range(3):w[:,:,oy+ty,ox+tx]+=wo[:,:,oy,ox]@wt[:,:,ty,tx]
        b=bo+wo.sum((-2,-1))@bt
        self.fused=nn.Conv2d(16,9,5,padding=2,device=linear.weight.device,dtype=linear.weight.dtype)
        with torch.no_grad():self.fused.weight.copy_(w);self.fused.bias.copy_(b)
    def phases(self,x):
        v=self.fused(x)
        if not self.edges:return v
        top=self.projection(self.tail(x[:,:,:3,:])[:,:,:2,:])[:,:,:1,:]
        bottom=self.projection(self.tail(x[:,:,-3:,:])[:,:,-2:,:])[:,:,-1:,:]
        left=self.projection(self.tail(x[:,:,:,:3])[:,:,:,:2])[:,:,:,:1]
        right=self.projection(self.tail(x[:,:,:,-3:])[:,:,:,-2:])[:,:,:,-1:]
        middle=torch.cat((left[:,:,1:-1,:],v[:,:,1:-1,1:-1],right[:,:,1:-1,:]),-1)
        return torch.cat((top,middle,bottom),-2)
    def forward(self,x):
        phases=self.phases(x).float().clamp(0,1)*255
        return F.interpolate(F.pixel_shuffle(phases,3),scale_factor=2,mode='nearest')

class ScaledPhaseOutput(nn.Module):
    def __init__(self,projection,factor,mode):
        super().__init__();self.conv=copy.deepcopy(projection);self.factor=factor;self.mode=mode
        if mode=='folded':
            with torch.no_grad():self.conv.weight.mul_(255);self.conv.bias.mul_(255)
    def forward(self,x):
        y=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))
        if self.mode=='folded':y=y.clamp(0,255)
        else:y=y.float().clamp(0,1)*255
        y=F.pixel_shuffle(y,self.factor)
        if self.factor!=6:y=F.interpolate(y,scale_factor=6//self.factor,mode='nearest')
        return y.float()

class ContinuationSystem(nn.Module):
    def __init__(self,base,output,display_units=False,space=False,skip_tail=False,project_after_resize=False):
        super().__init__();self.core=base.core;self.front=SpaceStatisticsFront(base.front,'space_pack') if space else base.front;self.output=output;self.display_units=display_units;self.skip_tail=skip_tail;self.project_after_resize=project_after_resize
    def forward(self,x,c):
        core,m=self.core,self.core.model;v=m.body(m.head(core.half_input(self.front(x))))
        r=m.global_reference.encode(core.half_input(c));r=r+r.mean((-2,-1),keepdim=True)
        if self.project_after_resize:r=m.global_reference.project(F.interpolate(r,size=v.shape[-2:],mode='bilinear',align_corners=False))
        else:r=F.interpolate(m.global_reference.project(r),size=v.shape[-2:],mode='bilinear',align_corners=False)
        v=v+r
        if not self.skip_tail:v=m.tail(v)
        value=self.output(v)
        return value if self.display_units else value.float().clamp(0,1)*255

def load_continuation(scene,case,run_dir,device='cuda'):
    run_dir=Path(run_dir)
    if case in ('baseline','previous_combo'):return load_structural(scene,'baseline' if case=='baseline' else 'space_pack_reference_relu_output_stable_phase3_quantsearch_nearest',run_dir,device=device)
    allowed=['relu_stats32_scaled36_exact','combo_relu_stats32_scaled9_exact','combo_aligned16_nearest','combo_aligned16_fixed','combo_aligned16_folded_fixed','combo_tail5_exact_edges','combo_tail5_interior','relu_stats24_scaled36_exact','combo_relu_stats24_scaled9_exact','native4_linear_dither_trained','native4_linear_dither_float_trained','native4_relu8_dither_trained','native4_relu8_ref_dither_trained','native4_linear_init','native4_linear_trained','native4_relu8_init','native4_relu8_trained','native4_relu8_ref_trained','fused9_3_channel','fused9_3_image','fused9_6_channel','scaled36_exact','scaled36_folded','scaled9_exact','scaled9_folded','combo_scaled9_exact','combo_scaled9_folded']
    allowed+=['combo_project_after_resize_scaled9_exact','project_after_resize_scaled36_exact','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact','combo_project_after_resize_aligned16_nearest','native4_linear_ref_dither_float_trained','combo_input9_half_aligned16_nearest','combo_input16_half_native_stats_aligned16_fixed','combo_aligned16_nearest_half_output','combo_aligned16_half_output_fixed']
    allowed+=['combo_stats3_scaled9_exact','combo_stats3_relu32_scaled9_exact','combo_input16_half_stats3_scaled9_exact','combo_stats3_aligned16_half_output_fixed','combo_stats3_relu32_aligned16_half_output_fixed']
    if case not in allowed:raise ValueError('Unknown continuation candidate '+case)
    ref=case.startswith('combo_') or '_ref_' in case
    base=load_structural(scene,'reference_relu_output_stable' if ref else 'baseline',run_dir,device=device)
    if case.startswith('native4'):
        kind='linear' if 'linear' in case else 'relu8';output=NativeFourOutput(base.core.model.upsample[0],kind)
        if case.endswith('trained'):
            ck=torch.load(run_dir/f'{scene}_native4_{kind}_004000.pt',map_location='cpu',weights_only=True);output.load_state_dict(ck['output'],strict=True)
        if 'dither' in case:output=DitherNativeOutput(output,'float' if 'dither_float' in case else 'fixed')
        return ContinuationSystem(base,output,space=True)
    if 'stats3_relu32' in case:base.front=RectifiedStatisticsFront(base.front,aligned=True,kernel_size=3)
    elif 'relu_stats' in case:base.front=RectifiedStatisticsFront(base.front,aligned='relu_stats32' in case)
    elif 'stats3' in case:base.front=PaddedInputFront(base.front,channels=16 if 'input16' in case else 9,kernel_size=3)
    elif 'input16' in case:base.front=PaddedInputFront(base.front)
    is9='9_' in case or 'tail5' in case or 'aligned16' in case
    if is9:
        projection=ReducedPhaseOutput(base.core.model.upsample[0],3,'preserve_nearest').conv
        ck=torch.load(run_dir/f'{scene}_output_quantsearch.pt',map_location='cpu',weights_only=True);projection.load_state_dict(ck['projection'],strict=True)
    else:projection=base.core.model.upsample[0]
    if 'aligned16' in case:
        model=ContinuationSystem(base,AlignedPhaseOutput(projection,case.split('aligned16_',1)[1]),display_units=True,space='input16' not in case and 'stats3' not in case,project_after_resize='project_after_resize' in case)
        model.input_channels=16 if 'input16' in case else 9;model.input_half='input9_half' in case or 'input16_half' in case
        return model
    if 'tail5' in case:return ContinuationSystem(base,TailFusedOutput(base.core.model.tail,projection,edges=case.endswith('exact_edges')),display_units=True,space=True,skip_tail=True)
    if case.startswith('fused9'):
        parts=case.split('_');output=FusedNineOutput(projection,int(parts[1]),parts[2]);return ContinuationSystem(base,output,space=True)
    model=ContinuationSystem(base,ScaledPhaseOutput(projection,3 if is9 else 6,'folded' if case.endswith('folded') else 'exact'),display_units=True,space='relu_stats' not in case and 'input16' not in case and 'stats3' not in case,project_after_resize='project_after_resize' in case)
    model.input_channels=16 if 'input16' in case else 9;model.input_half='input9_half' in case or 'input16_half' in case
    return model
