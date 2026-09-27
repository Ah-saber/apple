"""Independent engine controls of trained projection transplant and guards."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--run',type=Path);a=p.parse_args();base=Path(__file__).parent
sys.path[:0]=[str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages']
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
from reduce_output_phases import reduce_phases
run=a.run or Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-STRUCTURAL-20260927');folder=run/'source_onnx';rng=np.random.default_rng(927);rows=[]
cases=['phase3_quantsearch_nearest','phase3_quantsearch_fixed','space_pack_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_fixed']
for kind in ('constant','random','edge'):
 x=np.full((1,9,64,96),.3,np.float32);c=np.full((1,1,64,64),.3,np.float32)
 if kind=='random':x=rng.random(x.shape).astype(np.float32);c=rng.random(c.shape).astype(np.float32)
 if kind=='edge':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=.8
 inputs={'nine_raw':x,'reference_thumb':c}
 for case in cases:
  y=ReferenceEvaluator(str(folder/f'{a.scene}_{case}_small.onnx')).run(None,inputs)[0]
  z=ReferenceEvaluator(str(folder/f'{a.scene}_{case}_rewritten_small.onnx')).run(None,inputs)[0]
  d=np.abs(y.astype(np.float64)-z.astype(np.float64));assert d.max()<=.26,(case,kind,d.max())
  rows.append({'case':case,'input':kind,'mean_gray':float(d.mean()),'max_gray':float(d.max()),'bit_exact':bool(np.array_equal(y,z))})
old=onnx.load(folder/f'{a.scene}_baseline_small.onnx');native=onnx.load(folder/f'{a.scene}_baseline_small.onnx');projection=onnx.load(folder/f'{a.scene}_phase3_quantsearch_nearest_small.onnx')
init=next(i for i in native.graph.initializer if numpy_helper.to_array(i).shape==(36,16,3,3));w=numpy_helper.to_array(init).copy();w.flat[0]+=1;init.CopyFrom(numpy_helper.from_array(w,init.name))
try:reduce_phases(native,3,'preserve_nearest',projection,old)
except ValueError as e:assert 'differ' in str(e);guard=str(e)
else:raise AssertionError('Changed native projection accepted')
((run/'evidence' if (run/'evidence').is_dir() else run)/f'{a.scene}_searched_export_checks.json').write_text(json.dumps({'independent_engine':'ONNX ReferenceEvaluator','controls':rows,'changed_native_weight_rejected':guard,'SDK_verified':False},indent=2));print('PASSED',a.scene,len(rows),flush=True)
