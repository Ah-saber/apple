"""Replace fixed one-hot phase scatter with a rank-five reshape and transpose."""
import torch
from nativecontrast_phase11 import RawDenoiser

def scatter(phases):
    n,channels,height,width=phases.shape
    if channels!=4:raise ValueError('Four phase channels required')
    return phases.reshape(n,2,2,height,width).permute(0,3,1,4,2).reshape(n,1,height*2,width*2)

class StaticScatterDenoiser(RawDenoiser):
    def predict(self,raw):
        if self.encode.weight.dtype!=torch.float16:raise TypeError('Fitted inference branch must be FP16')
        latent=self.hidden(raw)
        phases=self.decode(latent)*(1/255)
        display=self.display_decode(latent)*(1/255)
        return (raw[:,-1:]+scatter(phases)).to(torch.float16),display

def convert(model):
    model.network.branch.__class__=StaticScatterDenoiser
    return model
