import copy,unittest
import torch
from torch import nn
from torch.nn import functional as F
from ir_sr.model import RT4KSRB0,staged_shuffle_permutation,to_deploy,inference_model
from ir_sr.auxiliary import training_model

class StudentV2Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(928)
 def test_all_shuffle_phases_and_lossless_packing(self):
  for factors in [(2,3),(2,2,3)]:
   scale=1
   for f in factors:scale*=f
   x=torch.arange(scale*scale*3*5).reshape(1,scale*scale,3,5).float()
   z=x[:,staged_shuffle_permutation(factors)]
   for f in factors:z=F.pixel_shuffle(z,f)
   torch.testing.assert_close(z,F.pixel_shuffle(x,scale),atol=0,rtol=0)
  raw=torch.randn(2,1,32,36)
  torch.testing.assert_close(F.pixel_shuffle(F.pixel_unshuffle(raw,4),4),raw,atol=0,rtol=0)
 def test_padding_crop_reference_and_fusion(self):
  for kernel in [1,3]:
   c=dict(channels=16,blocks=4,normalization='none',activation='relu',shuffle_mode='split223',packing_factor=4,output_kernel=kernel,auxiliary_raw_weight=.1,middle_gt_root='unused',global_reference=True)
   model=training_model(c)
   with torch.no_grad():model.global_reference.project.weight.normal_(std=.02)
   x=torch.randn(2,1,34,38);context=torch.randn(2,1,64,64);box=torch.tensor([[0.,0.,1.,1.],[.2,.1,.8,.9]])
   y,mid=model(x,return_auxiliary=True,context=context,context_box=box)
   self.assertEqual(y.shape,(2,1,102,114));self.assertEqual(mid.shape,x.shape)
   y0,x0,y1,x1=box.unbind(1);extended=torch.stack([y0,x0,y0+(y1-y0)*36/34,x0+(x1-x0)*40/38],1)
   manual=model(F.pad(x,(0,2,0,2),mode='reflect'),context=context,context_box=extended)[...,:102,:114]
   torch.testing.assert_close(y,manual,atol=0,rtol=0)
   (y.abs().mean()+mid.abs().mean()*.1).backward()
   self.assertGreater(float(model.global_reference.project.weight.grad.abs().sum()),0)
   opt=torch.optim.Adam(model.parameters(),lr=2e-4);opt.step()
   deploy=to_deploy(model)
   torch.testing.assert_close(model(x,context=context,context_box=box),deploy(x,context=context,context_box=box),atol=2e-5,rtol=2e-5)
   loaded=inference_model(c,model.state_dict())
   torch.testing.assert_close(model(x,context=context,context_box=box),loaded(x,context=context,context_box=box),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
