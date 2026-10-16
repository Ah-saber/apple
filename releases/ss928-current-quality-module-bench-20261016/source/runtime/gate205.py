"""Learn a bounded fraction of the current-small203 proposal above fixed198."""
from pathlib import Path
import torch,numpy as np
from body_current116 import CurrentRepair116

CACHE205=Path('/tmp/rawir-body205-20261014')
def features205(features,proposal):
    result=np.asarray(features,np.float32).copy();result[8]=np.clip(proposal/8,-4,4).astype(np.float16).astype(np.float32)
    return result

class Gate205(CurrentRepair116):
    def __init__(self):
        super().__init__()
        with torch.no_grad():self.net[-1].bias.fill_(0)
    def forward(self,features,proposal):return torch.sigmoid(self.net(features.clamp(-4,4)))*proposal
