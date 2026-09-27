"""Exercise reference/statistics/output composition on frozen source controls.
These are derived source graphs, never the inaccessible deployment native R2.
"""
import json
from pathlib import Path
import onnx
from rewrite_reference_relu import rewrite as reference
from rewrite_hotspots import rewrite_statistics
from transplant_output import replace_output
base=Path(__file__).resolve().parents[1];out=base/'compatibility_controls';out.mkdir(exist_ok=True);rows=[]
for scene in ('ordinary','special'):
 for suffix in ('','_small'):
  frozen=onnx.load(base/f'frozen_source/{scene}_baseline{suffix}.onnx');ref=onnx.load(base/f'source_onnx/{scene}_previous_combo{suffix}.onnx');native,count=reference(frozen,frozen,ref);native=rewrite_statistics(native,'space_pack')
  for case in ('combo_aligned16_nearest','combo_aligned16_fixed','combo_aligned16_folded_fixed','combo_aligned16_half_output_fixed','native4_linear_ref_dither_float_trained'):
   source=onnx.load(base/f'source_onnx/{scene}_{case}{suffix}.onnx');g,meta=replace_output(native,source,frozen);path=out/f'{scene}_{case}{suffix}.onnx'
   if path.exists():raise FileExistsError(path)
   onnx.save(g,path);rows.append({'file':path.name,'source_control':True,'actual_native_R2_verified':False,'reference_relu_count':count,**meta});print('CONTROL',path.name,flush=True)
(base/'evidence/composed_source_controls.json').write_text(json.dumps({'controls':rows,'SDK_verified':False},indent=2))
