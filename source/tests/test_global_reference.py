import unittest
import numpy as np
import torch
from torch import nn
from ir_sr.model import GlobalReference,RT4KSRB0,inference_model
from ir_sr.data import RawDisplayDataset,geometric_transform

class GlobalReferenceTests(unittest.TestCase):
 def test_zero_initialization_preserves_main_and_auxiliary(self):
  torch.manual_seed(928);old=RT4KSRB0(auxiliary_raw=True)
  torch.manual_seed(928);new=RT4KSRB0(auxiliary_raw=True,global_reference=True)
  for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,new.state_dict()[k]))
  x=torch.randn(2,1,24,24);context=torch.randn(2,1,64,64);box=torch.tensor([[.1,.2,.8,.9],[0,0,1,1]])
  a=old(x,return_auxiliary=True);b=new(x,return_auxiliary=True,context=context,context_box=box)
  for u,v in zip(a,b):self.assertTrue(torch.equal(u,v))
  b[0].square().mean().backward();self.assertGreater(float(new.global_reference.project.weight.grad.abs().sum()),0)
  with torch.no_grad():new.global_reference.project.weight-=.01*new.global_reference.project.weight.grad
  c=new(x,context=context,context_box=box);d=new(x,context=context+1,context_box=box)
  self.assertGreater(float((c-d).abs().max()),0)
  trained=inference_model({'channels':24,'blocks':4,'auxiliary_raw_weight':.1,'global_reference':True},new.state_dict())
  self.assertTrue(torch.equal(c,trained(x,context=context,context_box=box)))
  with self.assertRaises(ValueError):trained(x)

 def test_crop_alignment_all_eight_augmentations_and_roi_isolation(self):
  yy,xx=np.mgrid[:96,:120];raw=(100*yy+xx).astype(np.float32)
  ds=RawDisplayDataset.__new__(RawDisplayDataset);ds._raw=lambda r:raw;ds.normalization_for=lambda r:{'offset':0,'scale':10000}
  record={'train_roi_tlhw':[0,16,96,88]};crop=(12,24,60,60)
  module=GlobalReference(1);module.encoder=nn.Identity();module.project=nn.Identity()
  thumb,box=ds.context_for(record,crop,0);base=module(thumb[None],box[None],(10,10)).detach().numpy()[0]
  for op in range(8):
   thumb,box=ds.context_for(record,crop,op);got=module(thumb[None],box[None],(10,10)).detach().numpy()[0]
   self.assertTrue(np.allclose(got,geometric_transform(base,op),atol=2e-6))
  before=ds.context_for(record,crop,0)[0].clone();raw[:,:16]=1e6;raw[:,104:]=-1e6
  self.assertTrue(torch.equal(before,ds.context_for(record,crop,0)[0]))

if __name__=='__main__':unittest.main()
