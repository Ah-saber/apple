import torch
from torch.nn import functional as F
from speed_variants import ConvSample,LinearTail
from linear_tail_corrections import CorrectedLinearTail

torch.set_num_threads(2);torch.manual_seed(930)
for sy,sx in [(2,2),(4,5),(8,10)]:
    x=torch.rand(1,4,64,64)
    actual=ConvSample(4,64*sy,64*sx)(x)
    expected=F.interpolate(x,size=(64*sy,64*sx),mode='bilinear',align_corners=False)
    error=(actual-expected).abs()
    print('CONV_SAMPLE_MATH',sy,sx,float(error.max()),flush=True)
    assert float(error.max())<2e-5
tail=torch.nn.Conv2d(16,16,3,padding=1);projection=torch.nn.Conv2d(16,4,3,padding=1)
fused=LinearTail(tail,projection);x=torch.randn(1,16,32,40)
error=(fused(x)-projection(tail(x))).abs()
print('LINEAR_TAIL_BORDER_MATH',float(error.max()),flush=True);assert float(error.max())<2e-5
for height,width in [(3,3),(8,9),(32,40)]:
    x=torch.randn(1,16,height,width);actual=CorrectedLinearTail(tail,projection)(x)
    error=(actual-projection(tail(x))).abs()
    print('CORRECTED_TAIL_MATH',height,width,float(error.max()),flush=True);assert float(error.max())<2e-5
