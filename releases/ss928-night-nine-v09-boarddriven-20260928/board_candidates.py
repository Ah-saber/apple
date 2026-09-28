"""Board-profile-driven complete model variants; original v0.8 release is immutable."""
import copy,os,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
_here=Path(__file__).resolve().parent
_run=_here.parent if _here.name=='code_snapshot' else _here
_v08=Path(os.environ.get('RAWIR_V08_RELEASE',str((_run/'v08') if (_run/'v08').exists() else _run.parent/'ss928-night-nine-v08-extreme-20260928')))
sys.path.insert(0,str(_v08/'code_snapshot/runtime' if (_v08/'code_snapshot/runtime').is_dir() else _v08/'runtime'))
from deployment_candidates import load_deployment as load_v08,prepare_inputs,ReferenceMeanFirst,PaddedReference

BASE='cal_preserve_mean6_body3_native4_alignpixel'

class GroupedRowsOutput(nn.Module):
    """Exact complete 3x gray output, grouping 8..128 rows or one direct full convolution.

    Fixed dimensions intentionally prevent dynamic Shape/Gather/Concat export chains.
    """
    def __init__(self,old,rows=16,direct=False):
        super().__init__();assert rows in (8,16,32,64,128) and rows%2==0
        self.conv=old.conv;self.rows=rows;self.group=rows//2;self.direct=direct
        if direct and rows!=16:raise ValueError('Direct control uses 16 row grouping')
        if direct:
            kernel=torch.zeros(self.group*4,1,self.group*6,6,device=self.conv.weight.device,dtype=self.conv.weight.dtype)
            for fy in range(self.group):
                for py in range(2):
                    for px in range(2):
                        ch=fy*4+py*2+px
                        for sy in range(3):
                            for sx in range(3):kernel[ch,0,fy*6+py*3+sy,px*3+sx]=1
        else:
            kernel=torch.zeros(self.group*4,rows,3,6,device=self.conv.weight.device,dtype=self.conv.weight.dtype)
            for fy in range(self.group):
                for py in range(2):
                    for px in range(2):
                        ch=fy*4+py*2+px
                        for sy in range(3):
                            yy=fy*6+py*3+sy
                            for sx in range(3):kernel[ch,yy%rows,yy//rows,px*3+sx]=1
        self.register_buffer('kernel',kernel)
    def forward(self,x):
        phases=self.conv(x.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last))[:,:4].clamp(0,1)*255
        n,c,h,w=(int(v) for v in phases.shape)
        if h%self.group:raise ValueError('Feature height not divisible by row grouping')
        packed=phases.reshape(n,c,h//self.group,self.group,w).permute(0,3,1,2,4).reshape(n,c*self.group,h//self.group,w)
        blocked=F.conv_transpose2d(packed,self.kernel,stride=(self.group*6,6) if self.direct else (3,6))
        if self.direct:return blocked
        return blocked.permute(0,2,1,3).reshape(n,1,h*6,w*6)

class StudentReference(nn.Module):
    def __init__(self,old,depth=3):
        super().__init__();assert depth in (3,4);self.depth=depth;self.project=copy.deepcopy(old.project)
        device=self.project.weight.device;dtype=self.project.weight.dtype
        convs=[nn.Conv2d(1,16,3,padding=1,device=device,dtype=dtype),nn.ReLU(),nn.Conv2d(16,16,3,padding=4,dilation=4,device=device,dtype=dtype),nn.ReLU()]
        if depth==4:convs += [nn.Conv2d(16,16,3,padding=8,dilation=8,device=device,dtype=dtype),nn.ReLU()]
        convs += [nn.Conv2d(16,12,3,padding=12 if depth==3 else 16,dilation=12 if depth==3 else 16,device=device,dtype=dtype),nn.ReLU()]
        self.encoder=nn.Sequential(*convs)
    def encode(self,x):return self.encoder(x)

class ByteDisplayOutput(nn.Module):
    """Complete display frame as uint8; tests C1 Report/output bandwidth tradeoff."""
    def __init__(self,old):super().__init__();self.old=old
    def forward(self,x):return self.old(x).round().clamp(0,255).to(torch.uint8)

class FusedTailProjection(nn.Module):
    """Compose linear tail and four used projection channels; optional exact borders."""
    def __init__(self,tail,projection,edges=False):
        super().__init__();self.edges=edges
        if isinstance(tail,nn.Conv2d):linear=tail
        elif isinstance(tail,nn.Sequential) and len(tail)==2 and isinstance(tail[0],nn.Identity) and hasattr(tail[1],'rep_conv'):
            linear=tail[1].rep_conv
        else:raise ValueError('Unsupported tail for linear fusion')
        if linear.weight.shape!=(16,16,3,3) or projection.weight.shape!=(16,16,3,3):raise ValueError('Unexpected tail/projection shape')
        for layer in (linear,projection):
            if layer.padding!=(1,1) or layer.stride!=(1,1) or layer.groups!=1 or layer.dilation!=(1,1):raise ValueError('Unexpected convolution geometry')
        t=linear.weight.detach().cpu().double();p=projection.weight[:4].detach().cpu().double()
        w=torch.zeros((4,16,5,5),dtype=torch.float64)
        for py in range(3):
            for px in range(3):
                for ty in range(3):
                    for tx in range(3):w[:,:,py+ty,px+tx]+=p[:,:,py,px]@t[:,:,ty,tx]
        bias=projection.bias[:4].detach().cpu().double()+p.sum((-2,-1))@linear.bias.detach().cpu().double()
        self.fused=nn.Conv2d(16,4,5,padding=2,device=projection.weight.device,dtype=projection.weight.dtype)
        with torch.no_grad():self.fused.weight.copy_(w);self.fused.bias.copy_(bias)
        if edges:self.tail=copy.deepcopy(tail);self.projection=copy.deepcopy(projection)
    @property
    def weight(self):return self.fused.weight
    def forward(self,x):
        value=self.fused(x)
        if not self.edges:return value
        def original(s):return self.projection(self.tail(s))[:,:4]
        top=original(x[:,:,:3,:])[:,:,:1,:]
        bottom=original(x[:,:,-3:,:])[:,:,-1:,:]
        left=original(x[:,:,:,:3])[:,:,:,:1]
        right=original(x[:,:,:,-3:])[:,:,:,-1:]
        middle=torch.cat((left[:,:,1:-1,:],value[:,:,1:-1,1:-1],right[:,:,1:-1,:]),-1)
        return torch.cat((top,middle,bottom),-2)

def load_board(scene,case,run_dir,device='cuda'):
    run=Path(run_dir);stem=case;depth=None;refafter=False;meanfirst=False;pad16=False;byteoutput=False;tail5=None
    if stem.endswith('_u8'):byteoutput=True;stem=stem[:-3]
    for suffix,d in [('_refstudent3',3),('_refstudent4',4)]:
        if stem.endswith(suffix):depth=d;stem=stem[:-len(suffix)];break
    while True:
        hit=False
        for suffix in ('_refafter','_meanfirst','_refpad16','_tail5interior','_tail5edges'):
            if stem.endswith(suffix):
                if suffix=='_refafter':refafter=True
                elif suffix=='_meanfirst':meanfirst=True
                elif suffix=='_refpad16':pad16=True
                else:tail5=suffix=='_tail5edges'
                stem=stem[:-len(suffix)];hit=True;break
        if not hit:break
    if stem=='v08_c11':model=load_v08(scene,BASE+'_block16_full',_v08,device=device)
    elif stem=='v08_c04':model=load_v08(scene,BASE+'_outcascade',_v08,device=device)
    elif stem.startswith('rows') or stem.startswith('raw6_rows') or stem=='direct48':
        raw6=stem.startswith('raw6_');layout=stem[5:] if raw6 else stem
        rows=16 if layout in ('rows16static','direct48') else int(layout[4:])
        base='preserve_raw6_body3_native4_alignpixel' if raw6 else BASE
        model=load_v08(scene,base,_v08,device=device)
        model.output=GroupedRowsOutput(model.output,rows=rows,direct=layout=='direct48').to(device=device).eval()
    else:raise ValueError('Unknown board candidate '+case)
    owner=model.core.model
    if refafter:model.project_after_resize=True
    if pad16:owner.global_reference=PaddedReference(owner.global_reference).to(device=device,dtype=torch.float16).eval()
    if meanfirst:owner.global_reference=ReferenceMeanFirst(owner.global_reference).to(device=device,dtype=torch.float16).eval()
    if depth:
        student=StudentReference(owner.global_reference,depth).to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
        path=run/f'{scene}_reference_student{depth}.pt';ck=torch.load(path,map_location='cpu',weights_only=True);student.encoder.load_state_dict(ck['student'],strict=True);owner.global_reference=student.eval()
    if tail5 is not None:
        model.output.conv=FusedTailProjection(owner.tail,model.output.conv,edges=tail5).to(device=device,dtype=torch.float16,memory_format=torch.channels_last).eval()
        model.skip_tail=True
    if byteoutput:model.output=ByteDisplayOutput(model.output).eval()
    return model.eval()
