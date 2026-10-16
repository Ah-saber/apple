"""Bounded cleanup of newly vacated target positions above the current road result."""
from pathlib import Path
import torch
from small_temporal258 import SmallTemporal258
CACHE284=Path('/tmp/rawir-tail284-20261016')
class TailContext284(SmallTemporal258):
    def forward(self,features,mask):return 8*torch.tanh(self.net(features.clamp(-4,4)))*mask
