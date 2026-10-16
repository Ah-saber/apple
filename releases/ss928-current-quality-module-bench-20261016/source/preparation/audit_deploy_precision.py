"""Compare explicit FP32 boundaries with captured original GPU module outputs."""
import json
from pathlib import Path
import numpy as np
import torch
from export_current_deploy import construct, FixedCall

BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'


def main():
    modules, controller = construct('weather_heavy_c32')
    records = []
    with torch.inference_mode():
        for path in sorted((PACKAGE/'models'/'weather_heavy_c32').glob('graph_*/manifest.json')):
            manifest = json.loads(path.read_text())
            if not manifest.get('boundary_precision_changes'):
                continue
            folder = path.parent
            source_folder = BASE/'capture_archive'/folder.relative_to(PACKAGE)
            source = np.load(source_folder/'val_inputs.npz')
            names = source.files
            original = np.load(source_folder/'val_pytorch_output.npz')
            values = tuple(torch.from_numpy(source[name].astype(np.float32) if source[name].dtype == np.float64 else source[name]).cuda() for name in names)
            wrapper = FixedCall(modules[manifest['module_path']], manifest['argument_spec'], names)
            outputs = wrapper(*values)
            comparisons = []
            for index, output in enumerate(outputs):
                name = 'output_%03d'%index
                error = np.abs(output.float().cpu().numpy()-original[name].astype(np.float32))
                comparisons.append(dict(output=name, MAE=float(error.mean()), max_absolute=float(error.max())))
            manifest['GPU_boundary_conversion_comparison'] = comparisons
            path.write_text(json.dumps(manifest, indent=2))
            records.append(dict(graph=folder.name, frame=manifest['cases']['val']['frame'], comparisons=comparisons,
                max_input_magnitude=max(float(np.abs(source[item['input']]).max()) for item in manifest['boundary_precision_changes'])))
    result = dict(purpose='FP64 temporary interface to FP32; original captured GPU output compared after float32 cast',
        records=records, max_absolute=max(item['max_absolute'] for record in records for item in record['comparisons']),
        nonzero_interface_cases=sum(record['max_input_magnitude']>0 for record in records),
        training=False, SDK_compiled=False, full_pipeline_quality_check=False)
    (PACKAGE/'BOUNDARY_PRECISION.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='records'}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    main()
