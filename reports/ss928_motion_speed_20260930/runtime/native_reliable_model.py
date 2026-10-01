"""Native-grid spatial consistency controls temporal averaging before phase extraction."""
import torch
from torch.nn import functional as F

from reliable_model import ReliableModel


class NativeReliableModel(ReliableModel):
    def __init__(self,reference,width=16,threshold_gray=8.):
        super().__init__(reference,width,threshold_gray)
        kernel=torch.zeros(2,9,1,1)
        kernel[0,8]=1
        kernel[1,:8]=1/8
        self.register_buffer('temporal_kernel',kernel)

    def components(self,stack):
        dtype=self.front.weight.dtype
        pair=F.conv2d(stack.to(dtype),self.temporal_kernel.to(dtype))
        current=pair[:,:1];history=pair[:,1:]
        delta=history-current
        coherent=F.avg_pool2d(delta,3,1,1,count_include_pad=False)
        reliability=(1-coherent.abs()/(self.threshold_gray/255.)).clamp(0,1)
        contribution=reliability*delta
        fused=current+contribution
        phases=F.conv2d(fused,self.current_kernel.to(dtype),stride=4)
        return phases,F.avg_pool2d(current,4,4),F.avg_pool2d(contribution,4,4),reliability
