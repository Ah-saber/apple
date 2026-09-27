"""Compose bounded rewrites on the deployment-side v0.7 native graph."""
import argparse,hashlib,json
from pathlib import Path
import onnx
from rewrite_hotspots import rewrite_statistics,rewrite_output
from reduce_output_phases import reduce_phases
from rewrite_reference_relu import rewrite as reference_relu
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--statistics',choices=['space_pack','temporal_space_pack']);p.add_argument('--permutation',choices=['axis_horizontal','axis_vertical']);p.add_argument('--phase-mode',choices=['preserve_nearest','preserve_fixed']);p.add_argument('--old-source',type=Path);p.add_argument('--ref-source',type=Path);p.add_argument('--projection-source',type=Path);a=p.parse_args()
if not any([a.statistics,a.permutation,a.phase_mode,a.ref_source]):p.error('Choose at least one model rewrite')
if a.permutation and a.phase_mode:p.error('Choose either full36 axis permutation or nine-phase output')
if a.ref_source and not a.old_source:p.error('Reference rewrite requires frozen baseline source')
if a.projection_source and (not a.old_source or not a.phase_mode):p.error('Trained projection requires frozen source and phase mode')
if a.output.exists() or a.output.with_suffix('.candidate.json').exists():raise FileExistsError(a.output)
model=onnx.load(a.native)
if a.statistics:model=rewrite_statistics(model,a.statistics)
if a.ref_source:model,count=reference_relu(model,onnx.load(a.old_source),onnx.load(a.ref_source))
if a.permutation:model=rewrite_output(model,a.permutation)
if a.phase_mode:model=reduce_phases(model,3,a.phase_mode,onnx.load(a.projection_source) if a.projection_source else None,onnx.load(a.old_source) if a.projection_source else None)
onnx.checker.check_model(model);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,a.output)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report={'native_sha256':sha(a.native),'output_sha256':sha(a.output),'statistics':a.statistics,'permutation':a.permutation,'phase_mode':a.phase_mode,'reference_source_sha256':sha(a.ref_source) if a.ref_source else None,'projection_source_sha256':sha(a.projection_source) if a.projection_source else None,'actual_native_reference_resize_paths_preserved':True,'SDK_compiled':False,'NPU_measured':False}
a.output.with_suffix('.candidate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
