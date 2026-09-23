import unittest
import torch
from torch.nn import functional as F
from ir_sr.auxiliary import training_model
from ir_sr.model import to_deploy
class RawSkipTests(unittest.TestCase):
 def test_initialization_and_learned_correction(self):
  torch.set_num_threads(1)
  c=dict(channels=16,blocks=4,normalization='none',activation='relu',shuffle_mode='split223',packing_factor=4,output_kernel=1,auxiliary_raw_weight=.1,middle_gt_root='unused',global_reference=True)
  torch.manual_seed(928);a=training_model(c)
  torch.manual_seed(928);b=training_model(dict(c,raw_skip=True))
  for k,v in b.state_dict().items():
   if k.startswith('upsample.0.'):self.assertEqual(float(v.abs().max()),0)
   else:torch.testing.assert_close(v,a.state_dict()[k],atol=0,rtol=0)
  x=torch.randn(2,1,34,38);ctx=torch.randn(2,1,64,64);box=torch.tensor([[0.,0.,1.,1.],[.2,.1,.8,.9]])
  expected=F.interpolate(x,scale_factor=3,mode='bilinear',align_corners=False)
  y=b(x,context=ctx,context_box=box);torch.testing.assert_close(y,expected,atol=0,rtol=0)
  opt=torch.optim.Adam(b.parameters(),lr=2e-4)
  for _ in range(3):
   opt.zero_grad();b(x,context=ctx,context_box=box).square().mean().backward();opt.step()
  self.assertGreater(float(b.head[0].weight.grad.abs().sum()),0)
  torch.testing.assert_close(b(x,context=ctx,context_box=box),to_deploy(b)(x,context=ctx,context_box=box),atol=2e-5,rtol=2e-5)
if __name__=='__main__':unittest.main()
