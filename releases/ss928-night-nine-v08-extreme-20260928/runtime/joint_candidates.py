"""Learned output projection+upsampling composition and fused-front candidates."""
import torch
from torch import nn
from torch.nn import functional as F
from operator_candidates import FoldedStatisticsFront

class LearnedUpsample(nn.Module):
    def __init__(self, original, factor=2, precision='float16', bias_mode='channel'):
        super().__init__()
        if factor not in (2,3,6):raise ValueError('Factor must divide six')
        self.factor=factor;self.rest=6//factor;self.bias_mode=bias_mode
        old=original.weight.detach().float();bias=original.bias.detach().float()
        dtype=torch.float32 if precision=='float32' else torch.float16
        kernel=torch.zeros((17,self.rest**2,3*factor,3*factor),device=old.device,dtype=dtype)
        for dy in range(factor):
            for dx in range(factor):
                for sy in range(self.rest):
                    for sx in range(self.rest):
                        phase=(self.rest*dy+sy)*6+self.rest*dx+sx;dest=self.rest*sy+sx
                        for ky in range(3):
                            for kx in range(3):kernel[:16,dest,(2-ky)*factor+dy,(2-kx)*factor+dx]=old[phase,:,ky,kx].to(dtype)
                        kernel[16,dest,factor+dy,factor+dx]=bias[phase].to(dtype)
        if bias_mode=='image':
            kernel=kernel[:16]
            self.register_buffer('phase_bias',bias.reshape(1,1,6,6).repeat(1,1,512,640).to(dtype))
        self.register_buffer('weight',kernel.contiguous(memory_format=torch.channels_last))
    def forward(self,features):
        x=features.to(dtype=self.weight.dtype,memory_format=torch.channels_last)
        # Static constants on export; extra channel carries each original phase bias.
        if self.bias_mode=='channel':
            shape=(int(x.shape[0]),1,int(x.shape[2]),int(x.shape[3]))
            ones=torch.ones(shape,dtype=x.dtype,device=x.device)
            x=torch.cat((x,ones),1).contiguous(memory_format=torch.channels_last)
        y=F.conv_transpose2d(x,self.weight,stride=self.factor,padding=self.factor)
        if self.rest!=1:y=F.pixel_shuffle(y,self.rest)
        if self.bias_mode=='image':y=y+self.phase_bias
        return y.to(features.dtype)

class JointSystem(nn.Module):
    def __init__(self,original,front='original',factor=0,precision='float16',front_checkpoint=None,bias_mode='channel'):
        super().__init__();self.core=original.core
        self.front=(original.front if front=='original' else (BalancedResidualFront(original.front) if front=='balanced' else FoldedStatisticsFront(original.front,precision)))
        if front_checkpoint is not None:
            self.front.load_state_dict(front_checkpoint['front'],strict=True)
            self.front=self.front.to(device=original.front.first.weight.device,dtype=torch.float16,memory_format=torch.channels_last)
        self.output=LearnedUpsample(original.core.model.upsample[0],factor,precision,bias_mode) if factor else None
    def forward(self,stack,context):
        c,m=self.core,self.core.model
        features=m.body(m.head(c.half_input(self.front(stack))))
        ref=m.global_reference.encode(c.half_input(context));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True))
        ref=F.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False)
        features=m.tail(features+ref)
        gray=self.output(features) if self.output is not None else F.pixel_shuffle(m.upsample[0](features),6)
        return gray.float().clamp(0,1)*255

def load_joint(scene,case,run_dir,device='cuda'):
    import sys
    from pathlib import Path
    import os
    bundled=Path(__file__).resolve().parents[1]/'frozen_system_dependencies'/'ss928-night-nine-system-speed-20260926'
    pkg=Path(os.environ.get('RAWIR_SYSTEM_RELEASE_DIR',str(bundled if bundled.is_dir() else Path(__file__).resolve().parents[2]/'ss928-night-nine-system-speed-20260926')))
    if not pkg.is_dir():
        research=Path('/tmp/ss928_release_verification_20260926/ss928-night-nine-system-speed-20260926')
        if not research.is_dir():raise FileNotFoundError('Missing frozen system-speed dependency; set RAWIR_SYSTEM_RELEASE_DIR')
        pkg=research
    sys.path.insert(0,str(pkg/'runtime'))
    from load_system_release import load_release
    base,_=load_release(scene,'front_f32',device=device)
    quant_mode='per_output' if case.endswith('_w8po') else ('per_tensor' if case.endswith('_w8pt') else None)
    if quant_mode:case=case[:-5]
    if case=='baseline':return apply_weight8_front(base,quant_mode) if quant_mode else base
    front='balanced' if 'balanced' in case else ('fused' if case.startswith('front') or case.startswith('joint') else 'original')
    factor=0
    for f in (2,3,6):
        if f'up{f}' in case:factor=f
    precision='float32' if case.endswith('_f32') else 'float16'
    checkpoint=None
    if 'qat' in case:checkpoint=torch.load(Path(run_dir)/f'{scene}_fused_front_qat_003000.pt',map_location='cpu',weights_only=True)
    elif 'refined' in case:checkpoint=torch.load(Path(run_dir)/f'{scene}_fused_front_003000.pt',map_location='cpu',weights_only=True)
    result=JointSystem(base,front,factor,precision,checkpoint,'image' if 'image' in case else 'channel').eval()
    return apply_weight8_front(result,quant_mode) if quant_mode else result

@torch.no_grad()
def apply_weight8_front(model,mode):
    def quant(w):
        w=w.float();scale=(w.abs().amax(tuple(range(1,w.ndim)),keepdim=True) if mode=='per_output' else w.abs().max()).clamp_min(1e-9)/127
        return (w/scale).round().clamp(-127,127)*scale
    front=model.front
    if hasattr(front,'stats_weight'):
        front.stats_weight.copy_(quant(front.stats_weight));front.first.weight.copy_(quant(front.first.weight))
    elif hasattr(front,'temporal_weight'):
        front.temporal_weight.copy_(quant(front.temporal_weight));front.current_weight.copy_(quant(front.current_weight))
    else:front.weight.copy_(quant(front.weight))
    front.last.weight.copy_(quant(front.last.weight))
    return model

class BalancedResidualFront(nn.Module):
    """Separate current-frame features from temporal residual; aligned weight8 scale.
    Every temporal spatial tap has integer coefficient sum zero. Positive feature
    gains are absorbed into the final Conv through ReLU homogeneity.
    """
    def __init__(self,original):
        import copy
        import numpy as np
        super().__init__();width=original.first.weight.shape[0]
        folded=FoldedStatisticsFront(original,'float32')
        residual=folded.weight.detach().cpu().double().numpy().copy()
        current=np.zeros((width+4,1,6,6),np.float64)
        first=original.first.weight.detach().cpu().double().numpy()
        for dy in range(2):
            for dx in range(2):
                phase=2*dy+dx
                for ky in range(3):
                    for kx in range(3):
                        value=first[:,8+phase,ky,kx]
                        current[:width,0,2*ky+dy,2*kx+dx]=value
                        residual[:,8,2*ky+dy,2*kx+dx]-=value
                current[width+phase,0,2+dy,2+dx]=1
        # Remove tiny DC leakage from independently half-rounded statistics coefficients.
        residual[:,8]-=residual.sum(axis=1)
        maximum=np.max(np.abs(residual),axis=(1,2,3));global_max=float(maximum.max())
        if global_max<=0 or np.any(maximum<=0):raise ValueError('Degenerate temporal kernel')
        step=2.**np.floor(np.log2(global_max/127.))
        gain=127.*step/maximum
        ideal=residual*gain[:,None,None,None]/step
        q=np.clip(np.rint(ideal),-127,127).astype(np.int32)
        for c in range(width):
            for y in range(6):
                for x in range(6):
                    values=q[c,:,y,x];protected=np.abs(values)==127
                    while values.sum()!=0:
                        direction=-1 if values.sum()>0 else 1
                        candidates=[t for t in range(9) if not protected[t] and -127<=values[t]+direction<=127]
                        if not candidates:raise ValueError('Cannot balance temporal tap')
                        costs=[(values[t]+direction-ideal[c,t,y,x])**2-(values[t]-ideal[c,t,y,x])**2 for t in candidates]
                        t=candidates[int(np.argmin(costs))];values[t]+=direction
        if not np.all(q.sum(axis=1)==0) or not np.all(np.max(np.abs(q),axis=(1,2,3))==127):raise ValueError('Temporal coefficient proof failed')
        dtype=original.first.weight.dtype;device=original.first.weight.device
        temporal=torch.from_numpy((q*step).astype(np.float32)).to(device=device,dtype=dtype)
        if not torch.all(temporal.float().sum(1)==0):raise ValueError('Stored half coefficients lose zero-sum invariant')
        current[:width]*=gain[:,None,None,None]
        bias=np.concatenate((original.first.bias.detach().cpu().double().numpy()*gain,np.zeros(4)))
        self.register_buffer('temporal_weight',temporal.contiguous(memory_format=torch.channels_last))
        self.register_buffer('current_weight',torch.from_numpy(current.astype(np.float32)).to(device=device,dtype=dtype).contiguous(memory_format=torch.channels_last))
        self.register_buffer('current_bias',torch.from_numpy(bias.astype(np.float32)).to(device=device,dtype=dtype))
        self.last=copy.deepcopy(original.last)
        with torch.no_grad():self.last.weight.div_(torch.from_numpy(gain).to(device=device,dtype=dtype)[None,:,None,None])
        self.width=width;self.scale_step=float(step);self.feature_gains=gain.tolist()
    def forward(self,stack):
        x=stack.to(dtype=self.temporal_weight.dtype,memory_format=torch.channels_last)
        current=F.conv2d(x[:,-1:],self.current_weight,self.current_bias,stride=2,padding=2)
        temporal=F.conv2d(x,self.temporal_weight,stride=2,padding=2)
        feature=F.relu(current[:,:self.width]+temporal)
        return current[:,self.width:].float()+self.last(feature).float()*.025
