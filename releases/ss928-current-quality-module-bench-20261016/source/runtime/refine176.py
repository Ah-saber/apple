"""Bounded local refinement above frozen170, using independent physical features."""
from pathlib import Path
import cv2,numpy as np
from body_current116 import CurrentRepair116

CACHE176=Path('/tmp/rawir-body176-20261014')
DELTA176=Path('/tmp/rawir-residual176-20261014')

class Refine176(CurrentRepair116):
    def forward(self,features,mask):return super().forward(features,mask)/3

def features176(base,parent,physical_background,sr_background,physical,head,weight,current,past,gains):
    difference=physical_background-sr_background
    smooth=cv2.GaussianBlur(physical,(0,0),.8,borderType=cv2.BORDER_REFLECT)
    response=np.where(smooth>=0,gains['1'],gains['-1'])*smooth
    values=np.stack([base/255,parent/255,sr_background/255,physical_background/255,
        physical/16,response/16,difference/16,
        (difference-cv2.blur(difference,(101,101),borderType=cv2.BORDER_REFLECT))/16,
        head/24,(base-cv2.blur(base,(9,9),borderType=cv2.BORDER_REFLECT))/16,
        weight,current,past])
    return values.clip(-4,4).astype(np.float16)
