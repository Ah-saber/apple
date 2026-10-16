"""Use original-input body82, then the current frozen SR edge correction91."""
import numpy as np
import torch
from quality_c3283 import QualityC3283
from edge_reference91 import WIDTH91,mask91

class QualityC3292:
    def __init__(self,number=92):
        if number!=92:raise ValueError('This controller is round92')
        self.parent=QualityC3283(82)
        self.source='Original SR3 -> quality77 -> train-selected temporal body82 -> fixed current edge91'
    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        original={k:v for k,v in fields.items() if k!='shifted_current91'}
        source=self.parent.native(original,row,frame,initial=initial)
        image=source[0,0].float().cpu().numpy().copy()
        old=fields['current_only'][0,0,:,:WIDTH91].float().cpu().numpy()
        shifted=fields['shifted_current91'][0,0].float().cpu().numpy()
        image[:,:WIDTH91]+=np.clip(shifted-old,-8,8)*mask91()
        return torch.from_numpy(image.clip(0,255))[None,None].cuda()
