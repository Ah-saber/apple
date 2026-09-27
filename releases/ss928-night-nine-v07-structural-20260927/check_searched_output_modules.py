"""Independent output-path controls; prior full reference/statistics controls retained.
Cut precisely at learned output Conv input, retain all following computation.
"""
import copy,json,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
base=Path(__file__).parent;sys.path.insert(0,str(base/'tools'))
from reduce_output_phases import reduce_phases

def module(g):
 g=copy.deepcopy(g);ci={v.name:numpy_helper.to_array(v) for v in g.graph.initializer};matches=[n for n in g.graph.node if n.op_type=='Conv' and n.input[1] in ci and ci[n.input[1]].shape==(9,16,3,3)]
 if len(matches)!=1:raise ValueError('Unique nine-phase output required')
 cut=matches[0].input[0];lookup={o:n for n in g.graph.node for o in n.output};live=set()
 def visit(v):
  if v in live:return
  live.add(v)
  if v!=cut and v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 for v in g.graph.output:visit(v.name)
 nodes=[n for n in g.graph.node if any(o in live for o in n.output) and n.output[0]!=cut]
 # Tensor dimensions produced by source exporter are computed from this cut.
 graph=helper.make_graph(nodes,'isolated_final_projection',[helper.make_tensor_value_info(cut,TensorProto.FLOAT16,[1,16,3,5])],[helper.make_tensor_value_info(g.graph.output[0].name,TensorProto.FLOAT,[1,1,18,30])],[v for v in g.graph.initializer if v.name in live])
 m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);m.ir_version=8;onnx.checker.check_model(m);return m,cut
rng=np.random.default_rng(927);rows=[]
cases=['phase3_quantsearch_nearest','phase3_quantsearch_fixed','space_pack_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_fixed']
for scene in ['ordinary','special']:
 for case in cases:
  src,si=module(onnx.load(base/'source_onnx'/f'{scene}_{case}_small.onnx'));native,ni=module(onnx.load(base/'source_onnx'/f'{scene}_{case}_rewritten_small.onnx'))
  for kind in ['constant','random','edge']:
   x=np.full((1,16,3,5),.3,np.float16)
   if kind=='random':x=(rng.standard_normal(x.shape)*.3).astype(np.float16)
   if kind=='edge':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
   y=ReferenceEvaluator(src).run(None,{si:x})[0];z=ReferenceEvaluator(native).run(None,{ni:x})[0];d=np.abs(y.astype(np.float64)-z.astype(np.float64));assert d.max()<=.26,(scene,case,kind,d.max())
   rows.append({'scene':scene,'case':case,'input':kind,'mean_gray':float(d.mean()),'max_gray':float(d.max()),'bit_exact':bool(np.array_equal(y,z))})
 # Reject a native graph whose frozen Conv36 output weights no longer match.
 old=onnx.load(base/'source_onnx'/f'{scene}_baseline_small.onnx');changed=copy.deepcopy(old);projection=onnx.load(base/'source_onnx'/f'{scene}_phase3_quantsearch_nearest_small.onnx')
 init=next(i for i in changed.graph.initializer if numpy_helper.to_array(i).shape==(36,16,3,3));w=numpy_helper.to_array(init).copy();w.flat[0]+=1;init.CopyFrom(numpy_helper.from_array(w,init.name))
 try:reduce_phases(changed,3,'preserve_nearest',projection,old)
 except ValueError as e:assert 'differ' in str(e)
 else:raise AssertionError('Mismatched native weights accepted')
report={'scope':'isolated final projection through complete output path; prior full-graph statistics/reference controls are separate evidence','engine':'ONNX ReferenceEvaluator','controls':rows,'changed_native_weight_rejected_both_scenes':True,'full_new_combination_15_case_checks':'interrupted due slow local reference engine; not reported as passed','SDK_verified':False}
(base/'evidence/searched_output_module_checks.json').write_text(json.dumps(report,indent=2));print('PASSED',len(rows),'output controls and 2 frozen-weight guards',flush=True)
