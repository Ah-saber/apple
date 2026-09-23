"""Export equal-budget research candidates; CPU numerical parity, no board claims."""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,onnx,onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
from ir_sr.training import dataset_for_config,atomic_json,sha
from finalize_training import ExportGrayUnshuffle
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hierarchical-pooling',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--selection-label',default='fixed8000 equal-budget research candidate');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);start=time.monotonic();ckpt=a.checkpoint or a.run/'checkpoints/step_000008000.pt';s=torch.load(ckpt,map_location='cpu',weights_only=False);c=s['config'];step=s['progress']['step'];assert step>0
model=FullFrameStudent(inference_model(c,s['model']),hierarchical_pooling=a.hierarchical_pooling).eval();model.model.down=ExportGrayUnshuffle(model.model.packing_factor);del s
path=a.output/'student_raw_x3_1024x1280.onnx'
torch.onnx.export(model,torch.zeros(1,1,1024,1280),str(path),input_names=['raw'],output_names=['display'],opset_version=17,dynamo=False)
g=onnx.load(path);g=SymbolicShapeInference.infer_shapes(g,auto_merge=True);onnx.checker.check_model(g,full_check=True)
assert 'GridSample' not in [n.op_type for n in g.graph.node]
assert 'LayerNormalization' not in [n.op_type for n in g.graph.node]
assert not any('auxiliary_raw' in n.name for n in g.graph.initializer)
onnx.helper.set_model_props(g,{'input_contract':'FP32 NCHW 1x1x1024x1280; externally calibrated FRAME-SEQNORM-V1 normalized RAW, not old fixed DN normalization.',
 'output_contract':'FP32 NCHW 1x1x3072x3840; caller clips0..1, rint255 to uint8.',
 'calibration_sha256':c['sequence_normalization_sha256'],'reference':'Current input RAW avg_pool16x20 ->64x64 thumbnail; all reference processing inside graph; no GT.',
 'checkpoint_sha256':sha(ckpt),'status':a.selection_label+'; step='+str(step)+'. Offline normalization. SS928 UNTESTED.'})
onnx.save(g,path);options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
checks=[]
def check(name,x):
 with torch.inference_mode():gold=model(torch.from_numpy(x)).numpy()
 y=session.run(None,{'raw':x})[0];assert y.shape==(1,1,3072,3840) and np.isfinite(y).all()
 delta=(y-gold).astype(np.float64);record=dict(input=name,max_abs=float(np.abs(delta).max()),mae=float(np.abs(delta).mean()),rmse=float(np.sqrt((delta*delta).mean())))
 assert record['max_abs']<1e-3,record;checks.append(record);print(json.dumps(record),flush=True)
check('random_normal_seed928',np.random.default_rng(928).normal(size=(1,1,1024,1280)).astype(np.float32))
ds=dataset_for_config(c,'val')
for scene in c['scene_ids']:
 i=next(i for i,r in enumerate(ds.records) if r['scene_id']==scene);check(ds.records[i]['sample_id'],ds.full_raw(i)[None].numpy())
shapes={v.name:[d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim] for v in list(g.graph.input)+list(g.graph.value_info)+list(g.graph.output)}
shapes.update({v.name:list(v.dims) for v in g.graph.initializer})
def attribute(a):
 v=onnx.helper.get_attribute_value(a)
 if isinstance(v,onnx.TensorProto):return onnx.numpy_helper.to_array(v).tolist()
 if isinstance(v,bytes):return v.decode()
 return v
inventory=[dict(name=n.name,op=n.op_type,attributes={a.name:attribute(a) for a in n.attribute},inputs=[dict(name=v,shape=shapes.get(v)) for v in n.input],outputs=[dict(name=v,shape=shapes.get(v)) for v in n.output]) for n in g.graph.node]
atomic_json(a.output/'operator_inventory.json',inventory)
weights={v.name:list(v.dims) for v in g.graph.initializer};macs=[]
for node in g.graph.node:
 if node.op_type=='Conv':
  ws=weights[node.input[1]];ys=shapes[node.output[0]]
  macs.append(dict(name=node.name,macs=int(np.prod(ys)*np.prod(ws[1:]))))
atomic_json(a.output/'complexity.json',dict(conv_macs=sum(v['macs'] for v in macs),conv_layers=macs,parameter_count=sum(v.numel() for v in model.parameters()),float32_output_bytes=1*1*3072*3840*4,reference_resized_float32_bytes=c['channels']*(1024//c.get('packing_factor',2))*(1280//c.get('packing_factor',2))*4))
torch.save(dict(model=model.state_dict(),config=c,parent_checkpoint_sha256=sha(ckpt),deployment_options=dict(hierarchical_pooling=a.hierarchical_pooling)),a.output/'static_fused_weights.pt')
report=dict(status='passed_cpu_parity',hierarchical_pooling=a.hierarchical_pooling,source_checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),selection=a.selection_label,step=step,onnx_sha256=sha(path),opset=17,operator_counts=dict(Counter(n.op_type for n in g.graph.node)),checks=checks,seconds=time.monotonic()-start,SS928='not converted or measured; actual-shape AvgPool/Resize/reference graph compatibility pending',versions=dict(torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__))
atomic_json(a.output/'verification.json',report);print(json.dumps(report),flush=True)
