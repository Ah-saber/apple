"""Standalone checkpoint -> static ONNX, CPU parity, no training data required."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/src'))
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
class SpaceToDepth(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,factor):return torch.nn.functional.pixel_unshuffle(x,factor)
 @staticmethod
 def symbolic(g,x,factor):return g.op('SpaceToDepth',x,blocksize_i=factor)
class ExportDown(torch.nn.Module):
 def __init__(self,factor):super().__init__();self.factor=factor
 def forward(self,x):return SpaceToDepth.apply(x,self.factor)
p=argparse.ArgumentParser();p.add_argument('--group',choices=['day','light_medium','heavy','night'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
manifest=json.loads((ROOT/'MODEL_MANIFEST.json').read_text());identity=manifest['models'][a.group];ckpt=ROOT/'models'/a.group/'training_checkpoint.pt';assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==identity['checkpoint_sha256']
torch.set_num_threads(2);s=torch.load(ckpt,map_location='cpu',weights_only=False);assert s['progress']['step']==identity['step'];m=FullFrameStudent(inference_model(s['config'],s['model']),hierarchical_pooling=True).eval();m.model.down=ExportDown(s['config']['packing_factor']);del s
path=a.output/'model.onnx';torch.onnx.export(m,torch.zeros(1,1,1024,1280),str(path),input_names=['raw'],output_names=['display'],opset_version=17,dynamo=False);g=SymbolicShapeInference.infer_shapes(onnx.load(path),auto_merge=True);onnx.checker.check_model(g,full_check=True);onnx.save(g,path)
o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;rt=ort.InferenceSession(str(path),o,providers=['CPUExecutionProvider']);inputs=[('random_seed928',np.random.default_rng(928).normal(size=(1,1,1024,1280)).astype(np.float32))]
if a.input:inputs.append((str(a.input),np.fromfile(a.input,dtype='<f4').reshape(1,1,1024,1280)))
checks=[]
for name,x in inputs:
 assert np.isfinite(x).all()
 with torch.inference_mode():y=m(torch.from_numpy(x)).numpy()
 z=rt.run(['display'],{'raw':x})[0];assert np.isfinite(z).all();error=float(np.abs(z-y).max());assert error<1e-3
 checks.append({'input':name,'max_abs':error})
(a.output/'verification.json').write_text(json.dumps({'status':'passed','group':a.group,'checks':checks,'checkpoint_sha256':identity['checkpoint_sha256'],'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'note':'Different exporter versions may change graph bytes; new graph needs SS928 conversion and parity checks.'},indent=2));print(checks)
