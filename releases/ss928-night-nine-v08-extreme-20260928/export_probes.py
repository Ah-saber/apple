"""Full-shape module timing graphs, separate from complete-model results."""
import argparse,copy,fcntl,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-EXTREME-20260927';base=Path(__file__).parent
sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnx
from extreme_candidates import load_extreme,BASE_CASE
from continuation_candidates import load_continuation
from materialize_aliases import materialize
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2)
class HeadBody(nn.Module):
 def __init__(self,model):super().__init__();self.head=model.core.model.head;self.body=model.core.model.body
 def forward(self,x):return self.body(self.head(x))
class Reference(nn.Module):
 def __init__(self,reference):super().__init__();self.reference=reference
 def forward(self,x):
  r=self.reference.encode(x);return self.reference.project(r+r.mean((-2,-1),keepdim=True))
class Resize(nn.Module):
 def __init__(self,factor):super().__init__();self.factor=factor
 def forward(self,x):return F.interpolate(x,scale_factor=self.factor,mode='nearest')
class Fixed(nn.Module):
 def __init__(self,weight,stride):super().__init__();self.register_buffer('weight',weight);self.stride=stride
 def forward(self,x):return F.conv_transpose2d(x,self.weight,stride=self.stride)
class Relu(nn.Module):
 def forward(self,x):return F.relu(x)
parent=load_extreme(a.scene,BASE_CASE,run);middle=load_extreme(a.scene,'body2_quantsearch',run);npu=load_extreme(a.scene,'combo_stats3_relu32_aligned16_half_output_fixed',run)
probes=[('front_parent',parent.front,(1,9,1024,1280),torch.float16),('front_stats3_relu32',npu.front,(1,9,1024,1280),torch.float32),('head_body4',HeadBody(parent),(1,4,512,640),torch.float16),('head_body2',HeadBody(middle),(1,4,512,640),torch.float16),('reference_encode',Reference(parent.core.model.global_reference),(1,1,64,64),torch.float16),('reference_resize',nn.Upsample(size=(512,640),mode='bilinear',align_corners=False),(1,16,64,64),torch.float16),('tail',parent.core.model.tail,(1,16,512,640),torch.float16),('output_projection16',parent.output.conv,(1,16,512,640),torch.float16),('output_fixed6',Fixed(npu.output.kernel,6),(1,16,512,640),torch.float16),('output_resize2',Resize(2),(1,1,1536,1920),torch.float16),('output_resize3',Resize(3),(1,1,1024,1280),torch.float16),('full_half_relu',Relu(),(1,1,3072,3840),torch.float16)]
for tag in ('front3x1','front3x3','front5x1'):
 model=load_extreme(a.scene,tag+'_trained',run);probes.append((tag,model.front,(1,9,1024,1280),torch.float16))
for case in ('output_split3','output_native4_resize','output_native4_deconv'):
 model=load_extreme(a.scene,case,run);probes.append((case,model.output,(1,16,512,640),torch.float16))
out=run/'probes';out.mkdir(exist_ok=True);manifest=[]
for index,(tag,module,shape,dtype) in enumerate(probes):
 module=module.cuda().eval();x=torch.full(shape,.25,device='cuda',dtype=dtype);path=out/f'{a.scene}_{tag}.onnx'
 if path.exists():raise FileExistsError(path)
 with torch.inference_mode():y=module(x)
 torch.onnx.export(module,x,str(path),input_names=['probe_input'],output_names=['probe_output'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path)
 manifest.append({'file':path.name,'tag':tag,'input_shape':list(shape),'input_dtype':str(dtype),'output_shape':list(y.shape),'output_dtype':str(y.dtype),'input_constant_for_export_only':True,'runtime_input_seed':2795+index,'input_range':[-.25,.75],'full_model':False,'SDK_verified':False,'timing_includes_probe_input_format_and_report':True,'sum_not_equal_complete_model':True});print('PROBE',tag,flush=True)
(out/f'{a.scene}_manifest.json').write_text(json.dumps({'scene':a.scene,'probes':manifest},indent=2))
