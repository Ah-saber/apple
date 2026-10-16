"""Aligned causal RAW cues and bounded signed residual for current large bodies."""
from pathlib import Path
import numpy as np,torch
from raw_aligned269 import RawAligned269
from small_temporal258 import SmallTemporal258
CACHE275=Path('/tmp/rawir-large275-20261016')
CACHE277=Path('/tmp/rawir-large277-20261016')
class RawLarge275(RawAligned269):
    def amount(self,h,w):return float(np.clip((max(h,w)-36)/8,0,1))
class LargeTemporal275(SmallTemporal258):
    def forward(self,features,mask):return 8*torch.tanh(self.net(features.clamp(-4,4)))*mask
