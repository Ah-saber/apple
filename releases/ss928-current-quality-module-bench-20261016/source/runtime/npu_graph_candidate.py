"""Group fixed interpolation blocks; contract only proven linear night tails."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from speed_variants import LinearTail

class GroupedAxis(nn.Module):
    def __init__(self,axis):
        super().__init__()
        ranges=axis.ranges;pieces=[v.detach() for v in axis.weights]
        source=max(hi for lo,hi in ranges);width=max(hi-lo for lo,hi in ranges)
        outputs=[p.shape[0] for p in pieces]
        if len(set(outputs))!=1:raise ValueError('Grouped interpolation needs equal output blocks')
        self.windows=[(min(lo,source-width),min(lo,source-width)+width) for lo,hi in ranges]
        self.groups=len(pieces)
        weight=pieces[0].new_zeros(sum(outputs),width,1,1)
        begin=0
        for (lo,hi),(start,end),p in zip(ranges,self.windows,pieces):
            weight[begin:begin+p.shape[0],lo-start:hi-start]=p;begin+=p.shape[0]
        self.register_buffer('weight',weight)
        self.structure=dict(old_convolutions=self.groups,new_convolutions=1,
            old_output_concat_channels=sum(outputs),new_input_concat_channels=width*self.groups,
            old_coefficients=sum(p.numel() for p in pieces),new_coefficients=weight.numel(),groups=self.groups)
    def forward(self,value):
        packed=torch.cat([value[:,lo:hi] for lo,hi in self.windows],1)
        return F.conv2d(packed,self.weight,groups=self.groups)

def optimize(model,night_kind):
    candidate=copy.deepcopy(model);changes=[]
    def visit(parent,prefix=''):
        for name,child in list(parent.named_children()):
            label=prefix+'.'+name if prefix else name
            if child.__class__.__name__=='BlockAxis':
                if len(child.weights)<2:continue
                new=GroupedAxis(child)
                # C32 has a wide discontinuous window. Do not pad all groups to
                # that one width and multiply its dense cost across every block.
                if new.structure['new_coefficients']>new.structure['old_coefficients']*1.10:
                    changes.append(dict(module=label,grouped=False,reason='More than 10% coefficient padding; original axis retained',**new.structure))
                    continue
                setattr(parent,name,new);changes.append(dict(module=label,grouped=True,**new.structure))
            else:visit(child,label)
    visit(candidate)
    from shared_reference_axis11 import convert as share_reference_windows
    candidate,shared=share_reference_windows(candidate)
    changes.extend(dict(entry,weights_changed=False) for entry in shared)
    reference=candidate.reference if hasattr(candidate,'reference') and hasattr(candidate.reference,'kind') else candidate
    if night_kind:
        owner=reference.board.source.core.model;out=reference.board.source.output
        # Actual RepResBlock.forward is only rep_conv; body ReLU remains unchanged.
        merged=LinearTail(owner.tail,out.conv).cuda().half().eval()
        owner.tail=nn.Identity();out.conv=merged
        changes.append(dict(module='night_tail',old='16->16 k3, then 16->4 k3',new='16->4 k5 plus original four narrow borders',boundary_preserved=True,FP16_rounding_changed=True))
    return candidate.eval(),changes
