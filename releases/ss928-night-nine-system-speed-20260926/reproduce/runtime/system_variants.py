"""Whole-model variants; preserve nine frames and complete spatial output."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class FoldedFront(nn.Module):
 def __init__(self,old,precision='float32'):
  super().__init__();self.old=old;self.precision=precision
  w=old.stem.weight.detach().float();new=w[:,:9]*64.
  new[:,:6]-=w[:,:9].sum(1,keepdim=True)*(64./6.);new[:,8:9]+=w[:,9:10]
  self.register_buffer('weight',new if precision=='float32' else new.half());self.register_buffer('bias',old.stem.bias.detach().to(self.weight.dtype))
 def forward(self,stack):
  f=self.old;half=F.relu(F.conv2d(stack.to(dtype=self.weight.dtype,memory_format=torch.channels_last),self.weight,self.bias,stride=2,padding=2)).to(f.down.weight.dtype);x=F.relu(f.down(half))
  for layer in f.body:x=x+F.relu(layer(x))*.25
  correction=(f.output(x)+f.skip(half)).float()*.025
  return F.pixel_unshuffle(stack[:,-1:].float(),2)+correction

class QuarterFront(nn.Module):
 def __init__(self,width=8,blocks=2):
  super().__init__();self.width=width;self.blocks=blocks
  self.stem=nn.Conv2d(9,width,8,stride=4,padding=2)
  self.body=nn.ModuleList([nn.Conv2d(width,width,3,padding=1) for _ in range(blocks)])
  self.output=nn.ConvTranspose2d(width,4,4,stride=2,padding=1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,stack):
  x=F.relu(self.stem(stack.to(dtype=self.stem.weight.dtype,memory_format=torch.channels_last)))
  for layer in self.body:x=x+F.relu(layer(x))*.25
  return F.pixel_unshuffle(stack[:,-1:].float(),2)+self.output(x).float()*.025

class CompactReference(nn.Module):
 def __init__(self,width=8):
  super().__init__();self.width=width;self.first=nn.Conv2d(1,width,5,padding=2);self.last=nn.Conv2d(width,16,3,padding=1)
 def forward(self,context):
  x=self.last(F.relu(self.first(context.to(self.first.weight.dtype))))
  return x+x.mean((-2,-1),keepdim=True)

class TransposeOutput(nn.Module):
 """Reparameterize Conv36+shuffle6 into one learned transposed convolution.
 Extra constant feature carries the 36 phase-specific biases, without averaging.
 """
 def __init__(self,conv,factor=6):
  super().__init__();self.factor=factor;second=6//factor;self.second=second
  w=conv.weight.detach();bias=conv.bias.detach();kernel=3*factor
  weight=torch.zeros((17,second*second,kernel,kernel),device=w.device,dtype=w.dtype)
  for dy in range(factor):
   for dx in range(factor):
    for sy in range(second):
     for sx in range(second):
      channel=(second*dy+sy)*6+second*dx+sx;dest=sy*second+sx
      for ky in range(3):
       for kx in range(3):weight[:16,dest,(2-ky)*factor+dy,(2-kx)*factor+dx]=w[channel,:,ky,kx]
      weight[16,dest,factor+dy,factor+dx]=bias[channel]
  self.register_buffer('weight',weight.contiguous(memory_format=torch.channels_last))
 def forward(self,x):
  ones=torch.ones_like(x[:,:1]);x=torch.cat((x,ones),1).contiguous(memory_format=torch.channels_last)
  gray=F.conv_transpose2d(x,self.weight,stride=self.factor,padding=self.factor)
  if self.second!=1:gray=F.pixel_shuffle(gray,self.second)
  return gray

class FullSystem(nn.Module):
 def __init__(self,base,front=None,reference=None,output_mode='shuffle',output_dtype='float32'):
  super().__init__();self.core=base.core;self.front=front if front is not None else base.front;self.reference=reference;self.output_mode=output_mode;self.output_dtype=output_dtype
  self.output_layer=TransposeOutput(base.core.model.upsample[0],6 if output_mode=='transpose6' else 2) if output_mode.startswith('transpose') else None
 def forward(self,stack,context):
  c,m=self.core,self.core.model;features=m.body(m.head(c.half_input(self.front(stack))))
  if self.reference is None:
   ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
  else:ref=self.reference(context)
  ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False);features=m.tail(features+ref)
  gray=self.output_layer(features) if self.output_layer is not None else F.pixel_shuffle(m.upsample[0](features),6)
  gray=gray.float().clamp(0,1)*255
  if self.output_dtype=='float16':return gray.half()
  if self.output_dtype=='uint8':return gray.round().to(torch.uint8)
  return gray

class LowScaleSystem(nn.Module):
 """Clamp and scale the packed tensor before complete spatial expansion."""
 def __init__(self,base,front=None,reference=None,output_dtype='float32',byte_before_shuffle=False):
  super().__init__();self.core=base.core;self.front=front if front is not None else base.front;self.reference=reference;self.output_dtype=output_dtype;self.byte_before_shuffle=byte_before_shuffle
 def forward(self,stack,context):
  c,m=self.core,self.core.model;features=m.body(m.head(c.half_input(self.front(stack))))
  if self.reference is None:
   ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
  else:ref=self.reference(context)
  ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False);packed=m.upsample[0](m.tail(features+ref))
  if self.byte_before_shuffle:packed=(packed.float().clamp(0,1)*255).round().to(torch.uint8)
  else:packed=packed.clamp(0,1)*255
  gray=F.pixel_shuffle(packed,6)
  if self.byte_before_shuffle:return gray
  if self.output_dtype=='float16':return gray
  if self.output_dtype=='uint8':return gray.float().round().to(torch.uint8)
  return gray.float()

class PhaseRepeatSystem(nn.Module):
 """Derived RAW head plus half-precision repeat with nine fixed gray biases."""
 def __init__(self,base,front=None,reference=None,output_dtype='float32'):
  super().__init__();self.core=base.core;self.front=front if front is not None else base.front;self.reference=reference;self.output_dtype=output_dtype
  old=base.core.model.upsample[0];self.head=nn.Conv2d(16,4,3,padding=1).to(device=old.weight.device,dtype=old.weight.dtype,memory_format=torch.channels_last)
  with torch.no_grad():
   for dy in range(2):
    for dx in range(2):
     ids=[(3*dy+p)*6+3*dx+q for p in range(3) for q in range(3)];self.head.weight[dy*2+dx].copy_(old.weight[ids].float().mean(0));self.head.bias[dy*2+dx].copy_(old.bias[ids].float().mean(0))
  kernel=torch.zeros((4,9,2,2),device=old.weight.device,dtype=old.weight.dtype)
  for dy in range(2):
   for dx in range(2):kernel[dy*2+dx,:,dy,dx]=1
  self.register_buffer('repeat_weight',kernel.contiguous(memory_format=torch.channels_last));self.register_buffer('repeat_bias',torch.arange(-4,5,device=old.weight.device,dtype=old.weight.dtype)/9.)
 def forward(self,stack,context):
  c,m=self.core,self.core.model;features=m.body(m.head(c.half_input(self.front(stack))))
  if self.reference is None:
   ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
  else:ref=self.reference(context)
  ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
  packed=(self.head(m.tail(features+ref)).float().clamp(0,1)*255).half();phase=F.conv_transpose2d(packed,self.repeat_weight,self.repeat_bias,stride=2);gray=F.pixel_shuffle(phase,3).float().clamp(0,255)
  if self.output_dtype=='float16':return gray.half()
  if self.output_dtype=='uint8':return gray.round().to(torch.uint8)
  return gray

class CompactBody(nn.Module):
 def __init__(self,blocks=2):
  super().__init__();self.blocks=blocks;self.layers=nn.Sequential(*[layer for _ in range(blocks) for layer in (nn.Conv2d(16,16,3,padding=1),nn.ReLU())])
 def forward(self,x):return self.layers(x)

class FactorizedOutput(nn.Module):
 """Input-channel low-rank approximation to the output convolution."""
 def __init__(self,old,rank=8):
  super().__init__();self.rank=rank
  device,dtype=old.weight.device,old.weight.dtype
  self.first=nn.Conv2d(16,rank,1,bias=False).to(device=device,dtype=dtype,memory_format=torch.channels_last);self.last=nn.Conv2d(rank,36,3,padding=1).to(device=device,dtype=dtype,memory_format=torch.channels_last)
  with torch.no_grad():
   matrix=old.weight.float().permute(1,0,2,3).reshape(16,-1);u,s,vh=torch.linalg.svd(matrix,full_matrices=False);self.first.weight.copy_(u[:,:rank].T[:,:,None,None]);self.last.weight.copy_((s[:rank,None]*vh[:rank]).reshape(rank,36,3,3).permute(1,0,2,3));self.last.bias.copy_(old.bias);self.retained_energy=float(s[:rank].square().sum()/s.square().sum())
 def forward(self,x):return self.last(self.first(x))

class DirectHeadSystem(nn.Module):
 """Avoid current-frame pixel packing by an equivalent spatial head kernel.
 Splitting current and correction changes half-precision rounding and is checked.
 """
 def __init__(self,base,front=None,reference=None,low_scale=True,output_dtype='float32'):
  super().__init__();self.core=base.core;self.front=front if front is not None else base.front;self.reference=reference;self.low_scale=low_scale;self.output_dtype=output_dtype
  old=base.core.model.head[0];weight=torch.zeros((16,1,6,6),device=old.weight.device,dtype=old.weight.dtype)
  for dy in range(2):
   for dx in range(2):weight[:,0,dy::2,dx::2]=old.weight[:,dy*2+dx]
  self.register_buffer('current_weight',weight.contiguous(memory_format=torch.channels_last));self.register_buffer('current_bias',old.bias.detach().clone())
 def correction(self,stack):
  f=self.front
  if isinstance(f,QuarterFront):
   x=F.relu(f.stem(stack.to(dtype=f.stem.weight.dtype,memory_format=torch.channels_last)))
   for layer in f.body:x=x+F.relu(layer(x))*.25
   return f.output(x).float()*.025
  current=stack[:,-1:].float();residual=(stack.float()-stack[:,:6].float().mean(1,keepdim=True))*64.;raw=torch.cat((residual,current),1).to(dtype=f.stem.weight.dtype,memory_format=torch.channels_last);half=F.relu(f.stem(raw));x=F.relu(f.down(half))
  for layer in f.body:x=x+F.relu(layer(x))*.25
  return (f.output(x)+f.skip(half)).float()*.025
 def forward(self,stack,context):
  c,m=self.core,self.core.model;current=c.half_input(stack[:,-1:]);raw_features=F.conv2d(current,self.current_weight,self.current_bias,stride=2,padding=2);correction=c.half_input(self.correction(stack));corrected=F.conv2d(correction,m.head[0].weight,None,padding=1);features=m.body(raw_features+corrected)
  if self.reference is None:
   ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
  else:ref=self.reference(context)
  ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False);packed=m.upsample[0](m.tail(features+ref))
  gray=F.pixel_shuffle(packed.clamp(0,1)*255,6).float() if self.low_scale else F.pixel_shuffle(packed,6).float().clamp(0,1)*255
  if self.output_dtype=='float16':return gray.half()
  if self.output_dtype=='uint8':return gray.round().to(torch.uint8)
  return gray

class LinearFront(nn.Module):
 def __init__(self,kernel=2):
  super().__init__();self.kernel=kernel;self.conv=nn.Conv2d(9,4,kernel,stride=2,padding=(kernel-2)//2)
  with torch.no_grad():
   self.conv.weight.zero_();self.conv.bias.zero_();p=(kernel-2)//2
   for dy in range(2):
    for dx in range(2):self.conv.weight[dy*2+dx,8,p+dy,p+dx]=1.
 def forward(self,stack):return self.conv(stack.to(dtype=self.conv.weight.dtype,memory_format=torch.channels_last)).float()

class RowGroupedSystem(nn.Module):
 """Keep all output phases, regroup rows into channels before the same shuffle.
 Final reshape is inside the model and restores [N,1,3072,3840].
 """
 def __init__(self,base,front=None,reference=None,groups=16,output_dtype='float32',low_scale=False):
  super().__init__();self.core=base.core;self.front=front if front is not None else base.front;self.reference=reference;self.groups=groups;self.output_dtype=output_dtype;self.low_scale=low_scale
 def forward(self,stack,context):
  c,m=self.core,self.core.model;features=m.body(m.head(c.half_input(self.front(stack))))
  if self.reference is None:
   ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
  else:ref=self.reference(context)
  ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False);packed=m.upsample[0](m.tail(features+ref));b,channels,h,w=(int(v) for v in packed.shape);g=self.groups
  if self.low_scale:packed=packed.clamp(0,1)*255
  if b!=1:raise ValueError("Static SS928 row grouping requires batch one")
  grouped=packed.reshape(channels,g,h//g,w).permute(1,0,2,3).reshape(b,g*channels,h//g,w);gray=F.pixel_shuffle(grouped,6).reshape(b,1,h*6,w*6)
  if self.low_scale:
   if self.output_dtype=="float16":return gray
   if self.output_dtype=="uint8":return gray.round().to(torch.uint8)
   gray=gray.float()
  else:gray=gray.float().clamp(0,1)*255
  if self.output_dtype=='float16':return gray.half()
  if self.output_dtype=='uint8':return gray.round().to(torch.uint8)
  return gray

class StatsFront(nn.Module):
 """Compute current/old/recent packed statistics once, then learn correction."""
 def __init__(self,width=8):
  super().__init__();self.width=width;self.blocks=1
  weight=torch.zeros(12,9,2,2)
  for dy in range(2):
   for dx in range(2):
    phase=dy*2+dx;weight[phase,8,dy,dx]=1;weight[4+phase,:6,dy,dx]=1/6.;weight[8+phase,6:,dy,dx]=1/3.
  self.register_buffer('stats_weight',weight);self.first=nn.Conv2d(12,width,3,padding=1);self.last=nn.Conv2d(width,4,3,padding=1)
  nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
 def forward(self,stack):
  stats=F.conv2d(stack.to(dtype=self.stats_weight.dtype,memory_format=torch.channels_last),self.stats_weight,stride=2);correction=self.last(F.relu(self.first(stats))).float()*.025
  return stats[:,:4].float()+correction

class CenteredStatsFront(StatsFront):
 def forward(self,stack):
  stats=F.conv2d(stack.to(dtype=self.stats_weight.dtype,memory_format=torch.channels_last),self.stats_weight,stride=2);current=stats[:,:4].float();old=stats[:,4:8].float();recent=stats[:,8:12].float();features=torch.cat(((current-old)*64.,(current-recent)*64.,current),1).to(dtype=self.first.weight.dtype,memory_format=torch.channels_last);correction=self.last(F.relu(self.first(features))).float()*.025
  return current+correction

class SiLUReference(nn.Module):
 """Approximate GELU by x*sigmoid(beta*x), folding beta into convolutions."""
 def __init__(self,old,beta=1.702):
  super().__init__();self.beta=beta;self.ref=copy.deepcopy(old)
  with torch.no_grad():
   self.ref.encoder[0].weight.mul_(beta);self.ref.encoder[0].bias.mul_(beta);self.ref.encoder[2].bias.mul_(beta);self.ref.encoder[1]=nn.SiLU();self.ref.encoder[3]=nn.SiLU()
   if hasattr(self.ref,'pyramid'):
    for branch in self.ref.pyramid:branch[1].weight.mul_(beta);branch[1].bias.mul_(beta);branch[3].bias.mul_(beta);branch[2]=nn.SiLU()
   self.ref.project.weight.div_(beta)
 def forward(self,context):
  x=self.ref.encode(context.to(dtype=self.ref.project.weight.dtype,memory_format=torch.channels_last));return self.ref.project(x+x.mean((-2,-1),keepdim=True))

class BalancedCenteredFront(CenteredStatsFront):
 """Balanced temporal coefficients reduce constant-signal quantization drift.
 6-frame: 5*120/727 +127/727=1. 3-frame: 2*123/373 +127/373=1.
 """
 def __init__(self,width=8):
  super().__init__(width)
  for dy in range(2):
   for dx in range(2):
    p=dy*2+dx;self.stats_weight[4+p,:5,dy,dx]=120/727.;self.stats_weight[4+p,5,dy,dx]=127/727.;self.stats_weight[8+p,6:8,dy,dx]=123/373.;self.stats_weight[8+p,8,dy,dx]=127/373.

class InputLayoutSystem(nn.Module):
 """NHWC nine-frame input, with its tensor transpose retained in the model."""
 def __init__(self,model):
  super().__init__();self.model=model;self.output_dtype=model.output_dtype
 def forward(self,stack,context):return self.model(stack.permute(0,3,1,2),context)

class ContrastStatsFront(StatsFront):
 """Fold centered temporal differences into the fixed first convolution."""
 def __init__(self,width=8):
  super().__init__(width);self.stats_weight.zero_()
  for dy in range(2):
   for dx in range(2):
    p=dy*2+dx;self.stats_weight[p,8,dy,dx]=64.;self.stats_weight[p,:5,dy,dx]=-64*21/127.;self.stats_weight[p,5,dy,dx]=-64*22/127.;self.stats_weight[4+p,6:8,dy,dx]=-64*42/127.;self.stats_weight[4+p,8,dy,dx]=64*84/127.;self.stats_weight[8+p,8,dy,dx]=64.
 def forward(self,stack):
  stats=F.conv2d(stack.to(dtype=self.stats_weight.dtype,memory_format=torch.channels_last),self.stats_weight,stride=2);current=stats[:,8:].float()/64.;features=torch.cat((stats[:,:8],current.to(dtype=self.first.weight.dtype)),1);correction=self.last(F.relu(self.first(features))).float()*.025
  return current+correction
