"""Assemble the seven-scene component timing package without changing its models."""
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-ROUND5-20261007')
BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'
SCENES = ['night_ordinary', 'day_normal', 'weather_light', 'weather_medium', 'weather_heavy', 'weather_heavy_c32', 'night_special']


def main():
    for scene in SCENES[:4]:
        source = ROOT/'TIED500-DEPLOY-20261012-FIX'/scene
        target = PACKAGE/'models'/scene/'graph_base'
        target.mkdir(parents=True, exist_ok=True)
        original = json.loads((source/'manifest.json').read_text())
        graph = original['graphs']['tied500']
        for name, output in [('tied500.onnx', 'model.onnx'), ('smoke_inputs.npz', 'smoke_inputs.npz'), ('tied500_pc_output.npz', 'pc_output.npz')]:
            shutil.copy2(source/name, target/output)
        calibration = ['calibration_%02d.npz'%index for index in range(4)]
        for name in calibration:
            shutil.copy2(source/name, target/name)
        manifest = dict(id='graph_base', scene=scene, module_type='FrozenTiedSR',
            module_path='unchanged frozen SR', ONNX_exported=True, ONNX_file='model.onnx',
            inputs=graph['inputs'], outputs=graph['outputs'], operators=graph['operators'],
            smoke_inputs='smoke_inputs.npz', calibration_files=calibration, PC_case='test',
            unchanged_since_takeover=True, source=str(source), calibration=original['calibration'],
            original_PC_comparison=dict(MAE=graph['direct_PyTorch_vs_PC_MAE_gray'], max_absolute=graph['direct_PyTorch_vs_PC_max_gray']),
            checker_passed=graph['onnx_checker_passed'], SDK_compiled_for_this_package=False,
            current_board_measurement=False, full_pipeline=False)
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2))
        work = PACKAGE/'workloads'/scene
        work.mkdir(parents=True, exist_ok=True)
        (work/'val_frame_019.json').write_text(json.dumps(dict(scene=scene, split='test', frame=19,
            calls=['graph_base'], counts={'graph_base': 1}, full_pipeline=False), indent=2))
    graphs, summaries = [], []
    previous_comparisons = PACKAGE/'PC_COMPARISONS.json'
    comparisons = json.loads(previous_comparisons.read_text()) if previous_comparisons.exists() else []
    comparison_by_key = {(value['scene'], value['graph']): value for value in comparisons}
    for scene in SCENES:
        model_root = PACKAGE/'models'/scene
        scene_graphs = []
        for folder in sorted(model_root.glob('graph_*')):
            manifest_path = folder/'manifest.json'
            if not manifest_path.exists():
                raise RuntimeError('Unexported static graph '+str(folder))
            manifest = json.loads(manifest_path.read_text())
            calibration = manifest.get('calibration_files', ['train_graph_inputs.npz'])
            smoke_name = manifest.get('smoke_inputs', 'val_graph_inputs.npz')
            if not calibration or any(not (folder/name).is_file() for name in calibration) or not (folder/smoke_name).is_file():
                raise RuntimeError('Real calibration or smoke input missing '+str(folder))
            observed_path = folder/'val_pytorch_output.npz'
            if not observed_path.exists():
                observed_path = BASE/'capture_archive'/folder.relative_to(PACKAGE)/'val_pytorch_output.npz'
            if observed_path.exists():
                reference = np.load(observed_path)
                actual = np.load(folder/'pc_output.npz')
                delta = []
                for name in actual.files:
                    error = np.abs(actual[name].astype(np.float32)-reference[name].astype(np.float32))
                    delta.append(dict(output=name, MAE=float(error.mean()), max_absolute=float(error.max())))
                manifest['observed_GPU_vs_PC'] = delta
                manifest_path.write_text(json.dumps(manifest, indent=2))
                comparison_by_key[(scene, folder.name)] = dict(scene=scene, graph=folder.name, outputs=delta)
            entry = dict(key=scene+'/'+folder.name, scene=scene, id=folder.name,
                folder=str(folder.relative_to(PACKAGE)), module_type=manifest['module_type'],
                module_path=manifest['module_path'], inputs=manifest['inputs'], outputs=manifest['outputs'])
            graphs.append(entry)
            scene_graphs.append(entry)
        capture = model_root/'capture_manifest.json'
        unused = json.loads(capture.read_text())['modules_without_observed_call'] if capture.exists() else []
        plan = json.loads((PACKAGE/'workloads'/scene/'val_frame_019.json').read_text())
        summaries.append(dict(scene=scene, static_graphs=len(scene_graphs), calls_in_frame19=len(plan['calls']),
            unused_allocated_modules=unused, current_full_pipeline_NPU_ready=False))
    index = dict(name='current quality seven-scene neural-module timing package', date='2026-10-16',
        status='ONNX_and_real_cases_prepared_for_external_SDK', scenes=summaries, graphs=graphs,
        new_training=False, current_weights_unchanged=True,
        complete_frame_pipeline=False, CPU_state_tracking_and_fusion_ported=False,
        SDK_compiled=False, board_test_run=False,
        scope='Neural module output checks and NPU/SDK timing; explicit PC-prepared real tensor inputs',
        full_pipeline_timing_must_not_be_inferred_from_component_sum=True)
    (PACKAGE/'PACKAGE_INDEX.json').write_text(json.dumps(index, indent=2))
    (PACKAGE/'PC_COMPARISONS.json').write_text(json.dumps(list(comparison_by_key.values()), indent=2))
    # Preserve recorder-only argument arrays outside the upload package. Graph inputs remain.
    archive = BASE/'capture_archive'
    for pattern in ['val_inputs.npz', 'train_inputs.npz', 'val_pytorch_output.npz', 'train_pytorch_output.npz']:
        for source in list((PACKAGE/'models').glob('*/graph_*/'+pattern)):
            relative = source.relative_to(PACKAGE)
            target = archive/relative
            if not source.resolve().is_relative_to(PACKAGE.resolve()) or not target.resolve().is_relative_to(archive.resolve()):
                raise ValueError('Archive path outside task')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(source), str(target))
    inventory = [dict(file=str(path.relative_to(PACKAGE)), bytes=path.stat().st_size) for path in sorted(PACKAGE.rglob('*')) if path.is_file()]
    (PACKAGE/'FILE_SIZES.json').write_text(json.dumps(inventory, indent=2))
    print('PACKAGE_MODELS_READY', json.dumps(dict(graphs=len(graphs), scenes=summaries,
        bytes=sum(row['bytes'] for row in inventory), largest=sorted(inventory, key=lambda value:value['bytes'], reverse=True)[:5])), flush=True)


if __name__ == '__main__':
    main()
