"""Use actual display correction as the GT-error auxiliary component."""
import torch
from torch import nn
from spatiotime_phase11 import SpatialTime,CurrentRouteBoard
class TiedDecode(SpatialTime):
    def __init__(self,count):
        super().__init__(count)
        self.auxiliary=nn.Conv2d(8,4,1)
        with torch.no_grad():self.auxiliary.weight.zero_();self.auxiliary.bias.zero_()
    def evaluate(self,raw,auxiliary=False):
        def forward():
            shared=self.features(raw);correction=self.decode(shared)
            hints=torch.cat([self.auxiliary(shared).float(),correction.float()],1) if auxiliary else None
            return correction,hints
        if self.decode.weight.dtype==torch.float32:
            with torch.autocast(device_type='cuda',dtype=torch.float16):return forward()
        return forward()

