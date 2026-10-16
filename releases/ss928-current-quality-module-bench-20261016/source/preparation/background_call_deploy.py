"""Expose the live tensor-only background method as an ONNX module boundary."""
from torch import nn
from learned_background243 import LearnedBackground243


class BackgroundCall(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, features, masks):
        return LearnedBackground243.background(self.inner, features, masks)
