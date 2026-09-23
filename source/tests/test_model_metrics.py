import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'third_party/RT4KSR/code'))
from model.modules import ResBlock as OriginalResBlock, LayerNorm2d as OriginalNorm
from ir_sr.model import RT4KSRB0, ResBlock, LayerNorm2d, to_deploy
from ir_sr.metrics import image_metrics


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(928)
        torch.set_num_threads(1)

    def test_upstream_block_forward_and_gradients(self):
        for cls, original in ((ResBlock, OriginalResBlock), (LayerNorm2d, OriginalNorm)):
            ours, theirs = cls(24), original(24)
            ours.load_state_dict(theirs.state_dict(), strict=True)
            a = torch.randn(2, 24, 12, 14, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            y, z = ours(a), theirs(b)
            torch.testing.assert_close(y, z, atol=2e-6, rtol=2e-6)
            y.square().mean().backward()
            z.square().mean().backward()
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
            for p, q in zip(ours.parameters(), theirs.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)

    def test_parameter_counts_and_trained_fusion(self):
        model = RT4KSRB0()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 124740)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        x = torch.randn(2, 1, 16, 20)
        model(x).square().mean().backward()
        optimizer.step()
        deployed = to_deploy(model)
        self.assertEqual(sum(p.numel() for p in deployed.parameters()), 34980)
        for shape in ((1, 1, 14, 18), (2, 1, 32, 32)):
            x = torch.randn(shape)
            torch.testing.assert_close(model(x), deployed(x), atol=2e-5, rtol=2e-5)
            self.assertEqual(model(x).shape[-2:], (shape[-2] * 3, shape[-1] * 3))

    def test_metrics_against_independent_skimage(self):
        rng = np.random.default_rng(928)
        target = rng.random((48, 54), dtype=np.float32)
        prediction = np.clip(target + rng.normal(0, .03, target.shape), 0, 1).astype(np.float32)
        result = image_metrics(torch.tensor(prediction)[None, None], torch.tensor(target)[None, None])
        a, b = prediction[3:-3, 3:-3].astype(np.float64), target[3:-3, 3:-3].astype(np.float64)
        expected = structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=1.5,
                                         use_sample_covariance=False)
        self.assertAlmostEqual(result['ssim'], expected, places=9)
        self.assertAlmostEqual(result['psnr'], -10 * np.log10(np.mean((a-b)**2)), places=9)
        identical = image_metrics(torch.ones(1, 1, 32, 32) * .5, torch.ones(1, 1, 32, 32) * .5)
        self.assertEqual(identical['psnr'], 120)
        self.assertAlmostEqual(identical['ssim'], 1, places=10)


if __name__ == '__main__':
    unittest.main()
