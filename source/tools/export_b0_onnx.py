"""Export a completed run's fused B0 and verify CPU ORT on full-size inputs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ir_sr.model import RT4KSRB0
from ir_sr.training import atomic_json, sha, dataset_for_config
from finalize_training import ExportGrayUnshuffle


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Export audit is CPU-only'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    run = args.run
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    weights = run / 'artifacts/b0_deploy.pt'
    assert sha(weights) == result['deployment_weight_sha256']
    state = torch.load(weights, map_location='cpu', weights_only=False)
    assert state['parent_checkpoint_sha256'] == result['selected_checkpoint']['sha256']
    model = RT4KSRB0(config['channels'], config['blocks'], deploy=True).eval()
    model.load_state_dict(state['model'], strict=True)
    model.down = ExportGrayUnshuffle()
    destination = args.output / 'b0_gray_x3_1024x1280.onnx'
    torch.onnx.export(model, torch.zeros(1, 1, 1024, 1280), str(destination),
                      input_names=['raw'], output_names=['display'], opset_version=17, dynamo=False)
    graph = onnx.load(str(destination))
    root = Path(config['data_root'])
    recipe = json.loads((root / 'recipes/preprocess_v2.json').read_text())
    metadata = {
        'source_checkpoint_sha256': result['selected_checkpoint']['sha256'],
        'source_deploy_weight_sha256': sha(weights),
        'raw_normalization': json.dumps(recipe['raw_normalization']),
        'input_contract': 'float32 NCHW 1x1x1024x1280, pre-normalized RAW; no downsample at deployment',
        'output_contract': 'float32 NCHW 1x1x3072x3840; caller clips [0,1], multiplies 255 and rounds to uint8',
        'SS928_status': 'not converted or measured on SS928'}
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), options, providers=['CPUExecutionProvider'])
    trials = []

    def compare(name, array):
        reference = model(torch.from_numpy(array)).numpy()
        actual = session.run(None, {'raw': array})[0]
        assert actual.shape == (1, 1, 3072, 3840)
        difference = np.abs(reference - actual)
        record = {'input': name, 'max_abs_error': float(difference.max()),
                  'mean_abs_error': float(difference.mean()), 'tolerance': 1e-3}
        record['passed'] = record['max_abs_error'] < record['tolerance']
        trials.append(record)
        print(json.dumps(record), flush=True)
        assert record['passed'], record

    compare('random_normal_seed928', np.random.default_rng(928).normal(size=(1,1,1024,1280)).astype(np.float32))
    row = dataset_for_config(config, 'val').records[0]
    frame = np.load(root / row['raw']['path'], mmap_mode='r')[row['frame_id']].astype(np.float32)
    norm = recipe['raw_normalization']
    compare('native_raw:' + row['source_sample_id'], ((frame-norm['offset'])/norm['scale'])[None,None])
    report = {'status': 'passed', 'source_run': str(run), 'selected_checkpoint': result['selected_checkpoint'],
              'training_code_commit': json.loads((run/'training_identity.json').read_text())['code_commit'],
              'export_code_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'onnx_sha256': sha(destination), 'opset': 17,
              'operator_counts': dict(Counter(n.op_type for n in graph.graph.node)),
              'input_shape': [1,1,1024,1280], 'output_shape': [1,1,3072,3840], 'parity_trials': trials,
              'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
              'elapsed_seconds': time.monotonic()-started, 'SS928': 'not verified',
              'package_note': 'ONNX packages imported from TASK-019 isolated directory; restormer unchanged'}
    atomic_json(args.output / 'export_verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
