"""Full shape probes for reference, blocked report layout and quarter fusion."""
import argparse,fcntl,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');base=Path(__file__).parent;sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnx
from deployment_candidates import load_deployment,BlockedOutput
from materialize_aliases import materialize
lease=Path('/data/zhangbenzhuang/huawei_sr/runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2)
class Ref(nn.Module):
 def __init__(self,v):super().__init__();self.v=v
 def forward(self,x):
  v=self.v.encode(x);return self.v.project(v+v.mean((-2,-1),keepdim=True))
class Relu(nn.Module):
 def forward(self,x):return F.relu(x)
class Unpack(nn.Module):
 def forward(self,x):return x.permute(0,2,1,3).reshape(x.shape[0],1,x.shape[2]*16,x.shape[3])
probes=[]
for suffix in ('','_refpad16','_meanfirst','_refpad16_meanfirst'):
 model=load_deployment(a.scene,'cal_front3x1_trained_body2_native4_alignpixel'+suffix,r);probes.append(('reference_control'+(suffix or '_parent'),Ref(model.core.model.global_reference),(1,1,64,64),torch.float16))
q=load_deployment(a.scene,'quarter_w32_3x3_body1',r);qf=load_deployment(a.scene,'quarter_w32_3x3_body1_qat_fusedtail',r)
probes += [('quarter_front3x3',q.front,(1,9,1024,1280),torch.float16),('quarter_body32',q.body,(1,32,256,320),torch.float16),('quarter_tail32',q.tail,(1,32,256,320),torch.float16),('quarter_fused_tail_projection',qf.output.conv,(1,32,256,320),torch.float16),('quarter_reference_resize32',nn.Upsample(size=(256,320),mode='bilinear',align_corners=False),(1,32,64,64),torch.float16)]
for factor,model,shape in [(2,load_deployment(a.scene,'cal_front3x1_trained_body2_native4_alignpixel',r),(1,16,512,640)),(4,qf,(1,32,256,320))]:
 for split in (False,True):
  for full in (False,True):probes.append((f'output_factor{factor}_blocked_split{int(split)}_full{int(full)}',BlockedOutput(model.output,factor,full,split),shape,torch.float16))
for dtype in (torch.float16,torch.float32):
 tag='half' if dtype==torch.float16 else 'float32'
 probes += [('report_gray_'+tag,Relu(),(1,1,3072,3840),dtype),('report_block16_'+tag,Relu(),(1,16,192,3840),dtype),('block16_restore_'+tag,Unpack(),(1,16,192,3840),dtype),('input_nchw9_'+tag,Relu(),(1,9,1024,1280),dtype),('input_nhwc16_'+tag,Relu(),(1,1024,1280,16),dtype)]
out=r/'probes';rows=[]
for i,(tag,m,shape,dtype) in enumerate(probes):
 m=m.cuda().eval();x=torch.full(shape,.25,dtype=dtype,device='cuda');file=out/f'{a.scene}_{tag}.onnx'
 if file.exists():raise FileExistsError(file)
 with torch.inference_mode():y=m(x)
 torch.onnx.export(m,x,str(file),input_names=['probe_input'],output_names=['probe_output'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(file));onnx.checker.check_model(g);onnx.save(g,file);rows.append({'file':file.name,'tag':tag,'input_shape':list(shape),'input_dtype':str(dtype),'output_shape':list(y.shape),'output_dtype':str(y.dtype),'runtime_input_seed':2803+i,'input_range':[-.25,.75],'full_model':False,'SDK_verified':False,'timing_includes_probe_input_format_and_report':True,'sum_not_equal_complete_model':True});print('EXTRA_PROBE',tag,flush=True)
(out/f'{a.scene}_extra_manifest.json').write_text(json.dumps({'scene':a.scene,'probes':rows},indent=2))
