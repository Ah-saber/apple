"""Exercise rewrite on isolated real output modules, with bounded CPU controls."""
import copy,json,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
from onnx.reference import ReferenceEvaluator
from move_output_scale import move_scale
from materialize_aliases import materialize
base=Path(__file__).resolve().parents[1];old=base/'frozen_source';rng=np.random.default_rng(2709);rows=[]
def isolate(model):
 model,_=materialize(model);weights={v.name:numpy_helper.to_array(v) for v in model.graph.initializer};conv=next(n for n in model.graph.node if n.op_type=='Conv' and n.input[1] in weights and weights[n.input[1]].shape in [(9,16,3,3),(36,16,3,3)]);boundary=conv.input[0];end=model.graph.output[0].name;lookup={v:n for n in model.graph.node for v in n.output};live=set()
 def visit(v):
  if v in live:return
  live.add(v)
  if v!=boundary and v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 visit(end);nodes=[copy.deepcopy(n) for n in model.graph.node if any(o in live for o in n.output) and boundary not in n.output];init=[copy.deepcopy(i) for i in model.graph.initializer if i.name in live];channels=weights[conv.input[1]].shape[0];graph=helper.make_graph(nodes,'isolated_output',[helper.make_tensor_value_info(boundary,TensorProto.FLOAT16,[1,16,6,8])],[helper.make_tensor_value_info(end,TensorProto.FLOAT,[1,1,36,48])],init);m=helper.make_model(graph,opset_imports=list(model.opset_import));m.ir_version=model.ir_version;onnx.checker.check_model(m);return m,boundary,channels
for scene in ('ordinary','special'):
 for case in ('baseline','previous_combo'):
  source_path=(old if case=='baseline' else base/'source_onnx')/f'{scene}_{case}_small.onnx'
  original,input_name,channels=isolate(onnx.load(source_path));target=ReferenceEvaluator(original);exact=ReferenceEvaluator(move_scale(copy.deepcopy(original),'exact_phase'));folded=ReferenceEvaluator(move_scale(copy.deepcopy(original),'fold_phase'))
  for kind in ('constant','signed_random','edge'):
   x=np.full((1,16,6,8),.3,np.float16) if kind=='constant' else rng.normal(0,.3,(1,16,6,8)).astype(np.float16)
   if kind=='edge':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
   feed={input_name:x};a=target.run(None,feed)[0];b=exact.run(None,feed)[0];c=folded.run(None,feed)[0];assert np.array_equal(a,b)
   rows.append({'scene':scene,'phase_channels':channels,'input':kind,'exact_bit_equal':True,'folded_max_gray':float(np.abs(a-c).max()),'folded_mean_gray':float(np.abs(a-c).mean())})
(base/'evidence/output_scale_rewrite_controls.json').write_text(json.dumps({'controls':rows,'scope':'isolated frozen source output modules; CPU ONNX ReferenceEvaluator','SDK_verified':False},indent=2));print('PASSED',len(rows))
