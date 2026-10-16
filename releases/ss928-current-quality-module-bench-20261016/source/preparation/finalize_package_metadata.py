"""Record the finished package's file sizes and static I/O coverage."""
import json
from pathlib import Path

ROOT = Path('/tmp/rawir-current-deploy-20261016/package')
index = json.loads((ROOT/'PACKAGE_INDEX.json').read_text())
issues = []
for graph in index['graphs']:
    for value in graph['inputs']+graph['outputs']:
        if value['dtype'] not in ['FLOAT', 'FLOAT16'] or any(dimension <= 0 for dimension in value['shape']):
            issues.append(dict(graph=graph['key'], tensor=value))
    if not (ROOT/graph['folder']/'model.onnx').is_file():
        issues.append(dict(graph=graph['key'], missing='model.onnx'))
if issues:
    raise RuntimeError(json.dumps(issues))
log = (ROOT/'evidence'/'prepare_inputs.log').read_text()
prepared = sum(line.startswith('PREPARED ') for line in log.splitlines())
if prepared != len(index['graphs']):
    raise RuntimeError('Input preparation did not cover all static graphs')
inventory = [dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size) for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name not in ['FILE_SIZES.json', 'PACKAGE_SUMMARY.json']]
summary = dict(scenes=len(index['scenes']), static_graphs=len(index['graphs']), prepared_cases=prepared,
    files=len(inventory), bytes=sum(value['bytes'] for value in inventory),
    largest_file=max(inventory, key=lambda value:value['bytes']),
    SDK_compiled=False, ARM_runner_built=False, board_run=False,
    full_pipeline=False, ready_for='external SDK compilation and neural module measurements')
(ROOT/'FILE_SIZES.json').write_text(json.dumps(inventory, indent=2))
(ROOT/'PACKAGE_SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
