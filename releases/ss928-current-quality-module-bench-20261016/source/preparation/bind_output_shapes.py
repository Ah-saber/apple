"""Bind static output declarations to already computed PC output shapes."""
import json
from pathlib import Path
import shutil
import numpy as np
import onnx

BASE = Path('/tmp/rawir-current-deploy-20261016')
ROOT = BASE/'package'
index = json.loads((ROOT/'PACKAGE_INDEX.json').read_text())
changed = []
for entry in index['graphs']:
    folder = ROOT/entry['folder']
    manifest = json.loads((folder/'manifest.json').read_text())
    actual = np.load(folder/'pc_output.npz')
    if not any(value['shape'] != list(actual[value['name']].shape) for value in manifest['outputs']):
        continue
    archive = BASE/'symbolic_output_archive'/folder.relative_to(ROOT)
    if not archive.resolve().is_relative_to((BASE/'symbolic_output_archive').resolve()):
        raise ValueError('Archive outside task')
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder/'model.onnx', archive/'model.onnx')
    shutil.copy2(folder/'manifest.json', archive/'manifest.json')
    graph = onnx.load(folder/'model.onnx')
    for value in graph.graph.output:
        del value.type.tensor_type.shape.dim[:]
        for dimension in actual[value.name].shape:
            value.type.tensor_type.shape.dim.add().dim_value = int(dimension)
    onnx.checker.check_model(graph)
    onnx.save(graph, folder/'model.onnx')
    for value in manifest['outputs']:
        value['shape'] = list(actual[value['name']].shape)
    manifest['static_output_declarations_bound_to_PC_shape'] = True
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
    entry['outputs'] = manifest['outputs']
    changed.append(entry['key'])
(ROOT/'PACKAGE_INDEX.json').write_text(json.dumps(index, indent=2))
(ROOT/'STATIC_OUTPUT_SHAPES.json').write_text(json.dumps(dict(changed=changed, computation_changed=False), indent=2))
print('STATIC_OUTPUT_DECLARATIONS_BOUND', len(changed))
