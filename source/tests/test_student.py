import copy
import unittest
import torch
from torch import nn
from ir_sr.model import (RT4KSRB0, LayerNorm2d, to_deploy, use_shuffle23,
                         shuffle23_permutation, inference_model)
from ir_sr.auxiliary import training_model

class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(928)

    def test_phase_permutation_independent_coordinate_oracle(self):
        x = torch.arange(36*3*5).reshape(1,36,3,5).float()
        y = nn.PixelShuffle(3)(nn.PixelShuffle(2)(x[:,shuffle23_permutation()]))
        for r in range(18):
            for c in range(30):
                self.assertEqual(y[0,0,r,c], x[0,(r%6)*6+c%6,r//6,c//6])

    def test_shuffle_preserves_output_and_input_gradients(self):
        a = RT4KSRB0(auxiliary_raw=True,global_reference=True)
        with torch.no_grad():
            a.global_reference.project.weight.normal_(std=.01)
        b = use_shuffle23(copy.deepcopy(a))
        args = dict(context=torch.randn(2,1,64,64),context_box=torch.tensor([[0.,0.,1.,1.],[.2,.1,.7,.9]]))
        x = torch.randn(2,1,20,26,requires_grad=True)
        z = x.detach().clone().requires_grad_()
        ya, ma = a(x,return_auxiliary=True,**args)
        yb, mb = b(z,return_auxiliary=True,**args)
        torch.testing.assert_close(ya,yb,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(ma,mb,atol=0,rtol=0)
        ya.square().mean().backward();yb.square().mean().backward()
        torch.testing.assert_close(x.grad,z.grad,atol=2e-7,rtol=2e-5)

    def test_shared_initial_weights_and_strict_roundtrip_fusion(self):
        torch.manual_seed(928);baseline=RT4KSRB0(auxiliary_raw=True,global_reference=True)
        reference = use_shuffle23(copy.deepcopy(baseline)).state_dict()
        for activation in ('gelu','relu'):
            cfg=dict(channels=24,blocks=4,auxiliary_raw_weight=.1,middle_gt_root='unused',
                     global_reference=True,normalization='none',activation=activation,shuffle_mode='split23')
            torch.manual_seed(928);m=training_model(cfg)
            self.assertFalse(any(isinstance(v,LayerNorm2d) for v in m.modules()))
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,reference[k]),k)
            x=torch.randn(2,1,24,30);args=dict(context=torch.randn(2,1,64,64),context_box=torch.tensor([[0.,0.,1.,1.],[.1,.2,.8,.9]]))
            opt=torch.optim.Adam(m.parameters(),lr=2e-4)
            y,mid=m(x,return_auxiliary=True,**args)
            (y.abs().mean()+.1*mid.abs().mean()).backward();opt.step()
            inference=inference_model(cfg,m.state_dict()).eval()
            fused=to_deploy(m)
            self.assertFalse(hasattr(fused,'auxiliary_raw'))
            torch.testing.assert_close(m(x,**args),inference(x,**args),atol=0,rtol=0)
            torch.testing.assert_close(m(x,**args),fused(x,**args),atol=2e-5,rtol=2e-5)
            self.assertEqual(sum(isinstance(v,nn.GELU) for v in m.body.modules()),4 if activation=='gelu' else 0)

if __name__=='__main__':unittest.main()
