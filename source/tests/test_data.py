import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from ir_sr.data import area_downsample3, paired_patch


class DataTests(unittest.TestCase):
    def test_integer_grid_and_fractional_dn(self):
        raw = np.arange(36, dtype=np.uint16).reshape(6, 6)
        np.testing.assert_array_equal(area_downsample3(raw), [[7, 10], [25, 28]])
        spike = np.zeros((6, 6), dtype=np.uint16)
        spike[0, 0] = 1
        self.assertAlmostEqual(float(area_downsample3(spike)[0, 0]), 1 / 9, places=7)

    def test_synchronized_crop_and_no_source_mutation(self):
        raw = np.arange(216, dtype=np.uint16).reshape(12, 18)
        gt = raw.astype(np.uint8)
        original = raw.copy()
        x, y = paired_patch(raw, gt, 3, 6, 6, 6, 50, 10)
        self.assertEqual(x.shape, (1, 2, 2))
        self.assertAlmostEqual(float(x[0, 0, 0]), (79 - 50) / 10, places=6)
        np.testing.assert_array_equal(np.rint(y[0] * 255), gt[3:9, 6:12])
        np.testing.assert_array_equal(raw, original)

    def test_reject_invalid_geometry(self):
        raw = np.zeros((12, 18), dtype=np.uint16)
        with self.assertRaises(ValueError):
            area_downsample3(raw[:, :17])
        with self.assertRaises(ValueError):
            paired_patch(raw, raw, 10, 0, 6, 6, 0, 1)
        with self.assertRaises(ValueError):
            paired_patch(raw, raw, 0, 0, 9, 6, 0, 1)


if __name__ == '__main__':
    unittest.main()
