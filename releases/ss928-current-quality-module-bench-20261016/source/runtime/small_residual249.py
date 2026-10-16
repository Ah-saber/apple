"""Learn only the remaining compact-target error above the retained complete output."""
from pathlib import Path
import torch
from torch import nn
from small_wide244 import SmallWide244

CACHE249=Path('/tmp/rawir-small249-20261016')
CACHE253=Path('/tmp/rawir-small253-20261016')

class SmallResidual249(SmallWide244):
    def initialize_residual(self,state):
        self.load_state_dict(state)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,mask):return 16*torch.tanh(self.net(features.clamp(-4,4)))*mask
