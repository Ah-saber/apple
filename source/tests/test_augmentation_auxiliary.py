import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from ir_sr.data import RawDisplayDataset, area_downsample3, geometric_transform
from ir_sr.model import RT4KSRB0, inference_model, to_deploy
from ir_sr.auxiliary import loss_terms


class AugmentationTests(unittest.TestCase):
    def test_eight_transforms_preserve_values_and_area_geometry(self):
        raw = np.arange(144, dtype=np.float32).reshape(12, 12) + .125
        variants = [geometric_transform(raw, k) for k in range(8)]
        self.assertEqual(len({a.tobytes() for a in variants}), 8)
        for k, result in enumerate(variants):
            np.testing.assert_array_equal(np.sort(result.ravel()), np.sort(raw.ravel()))
            np.testing.assert_allclose(area_downsample3(result), geometric_transform(area_downsample3(raw), k), atol=1e-5)
            self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(variants[1], np.rot90(raw))
        np.testing.assert_array_equal(variants[4], raw[:, ::-1])

    def test_real_dataset_triple_alignment_epoch_workers_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ('recipes', 'manifests/split_s1', 'manifests/dataset_d1', 'middle'):
                (root / path).mkdir(parents=True)
            recipe = {'train_crop_hr': 12, 'validation_crop_tlhw': [0,0,12,12],
                      'raw_normalization': {'offset': 0, 'scale': 1}}
            (root / 'recipes/preprocess_v2.json').write_text(json.dumps(recipe))
            raw = np.arange(18*24, dtype=np.uint16).reshape(18,24)
            np.save(root/'raw.npy', raw[None])
            Image.fromarray((raw % 256).astype(np.uint8)).save(root/'target.png')
            roi = [0,6,18,12]
            np.save(root/'middle/dn.npy', (raw[:,6:18].astype(np.float32)+.25)[None])
            record = {'sample_id': 'sample', 'source_sample_id': 'source', 'scene_id': 'weather_heavy',
                      'domain':'adverse', 'sequence_id':'s', 'frame_id':0, 'evaluation_scope':'training',
                      'train_roi_tlhw':roi, 'raw':{'path':'raw.npy'}, 'target':{'path':'target.png'}}
            (root/'manifests/dataset_d1/pairs.jsonl').write_text(json.dumps(record)+'\n')
            for split in ('train','val','test'):
                (root/f'manifests/split_s1/{split}.txt').write_text('sample\n')
            (root/'middle/index.json').write_text(json.dumps({'status':'complete','dataset':'dataset_d1','sequences':[
                {'key':'adverse/s','path':'dn.npy','shape':[1,18,12], 'roi_tlhw':roi,
                 'frame_ids':[0],'sample_ids':['sample'],'source_raw_path':'raw.npy'}]}))
            ds = RawDisplayDataset(root, dataset_version='dataset_d1', geometric_augmentation=True,
                                   middle_gt_root=root/'middle')
            seen = set()
            for epoch in range(32):
                ds.set_epoch(epoch); item = ds[0]; again = ds[0]
                seen.add(item['augmentation_id'])
                torch.testing.assert_close(item['raw'], again['raw'], rtol=0, atol=0)
                torch.testing.assert_close(item['middle_raw'], item['raw']+.25, rtol=0, atol=1e-4)
                t,l,h,w = item['crop_tlhw']
                self.assertTrue(l >= 6 and l+w <= 18 and t+h <= 18)
                expected = geometric_transform((raw[t:t+h,l:l+w] % 256).astype(np.float32)/255, item['augmentation_id'])
                np.testing.assert_array_equal(item['gt'][0], expected)
            self.assertEqual(len(seen), 8)
            ds.set_epoch(7)
            batch = next(iter(DataLoader(ds, batch_size=1, num_workers=2)))
            torch.testing.assert_close(batch['raw'][0], ds[0]['raw'], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                RawDisplayDataset(root, 'val', dataset_version='dataset_d1', geometric_augmentation=True)
            validation = RawDisplayDataset(root, 'val', dataset_version='dataset_d1')
            self.assertEqual(validation[0]['augmentation_id'], 0)
            self.assertNotIn('middle_raw', validation[0])
            # Unobserved pixels cannot affect the cropped sample or cached middle label.
            before = ds[0]
            changed = raw.copy(); changed[:,:6]=65000; changed[:,18:]=65000
            np.save(root/'raw.npy', changed[None]); ds.cache.clear()
            torch.testing.assert_close(before['raw'], ds[0]['raw'], rtol=0, atol=0)


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(928)

    def test_training_head_gradients_and_inference_removal(self):
        model = RT4KSRB0(8,2,auxiliary_raw=True)
        raw = torch.randn(2,1,12,12)
        target = torch.randn(2,1,36,36)
        pred, mid = model(raw, return_auxiliary=True)
        torch.testing.assert_close(mid, raw, rtol=0, atol=0)
        # Nonzero head verifies the auxiliary gradient reaches the shared backbone.
        torch.nn.init.normal_(model.auxiliary_raw[0].weight, std=.01)
        model(raw, return_auxiliary=True)[1].square().mean().backward()
        self.assertGreater(float(model.head[0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.upsample[0].weight.grad)
        self.assertIsNone(model.tail[1].expand_conv.weight.grad)
        model.zero_grad()
        total, display, middle = loss_terms(model,raw,target,raw-.01,.1)
        torch.testing.assert_close(total,display+.1*middle)
        total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        plain = inference_model({'channels':8,'blocks':2,'auxiliary_raw_weight':.1}, model.state_dict()).eval()
        self.assertFalse(any('auxiliary' in k for k in plain.state_dict()))
        torch.testing.assert_close(model(raw), plain(raw), rtol=0, atol=0)
        deployed = to_deploy(model)
        self.assertFalse(any('auxiliary' in k for k in deployed.state_dict()))
        torch.testing.assert_close(model(raw), deployed(raw), rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            inference_model({'channels':8,'blocks':2}, model.state_dict())

    def test_disabled_head_preserves_seeded_baseline(self):
        torch.manual_seed(928); old = RT4KSRB0(8,2)
        torch.manual_seed(928); new = RT4KSRB0(8,2,auxiliary_raw=True)
        raw = torch.randn(1,1,12,12)
        torch.testing.assert_close(old(raw), new(raw), rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'ONNX tools unavailable')
    def test_export_has_one_output_and_no_auxiliary_parameters(self):
        import onnx
        import onnxruntime as ort
        model = RT4KSRB0(8,2,auxiliary_raw=True).eval()
        torch.nn.init.normal_(model.auxiliary_raw[0].weight, std=.01)
        plain = inference_model({'channels':8,'blocks':2,'auxiliary_raw_weight':.1},model.state_dict()).eval()
        deployed = to_deploy(plain)
        raw = torch.randn(1,1,12,12)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.onnx'
            torch.onnx.export(deployed,raw,path,input_names=['raw'],output_names=['display'],opset_version=17,dynamo=False)
            graph = onnx.load(str(path)); onnx.checker.check_model(graph)
            self.assertEqual(len(graph.graph.input),1); self.assertEqual(len(graph.graph.output),1)
            self.assertFalse(any('auxiliary' in x.name for x in graph.graph.initializer))
            options=ort.SessionOptions();options.intra_op_num_threads=2
            session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
            actual=session.run(None,{'raw':raw.numpy()})[0]
            with torch.no_grad():expected=plain(raw).numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)


if __name__ == '__main__':
    unittest.main()
