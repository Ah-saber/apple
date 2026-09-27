"""Actual preserved-front and split-reference dynamic module probes."""
import argparse,fcntl,json,sys
from pathlib import Path
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');c=Path(__file__).parent;sys.path[:0]=[str(c/'runtime'),str(c/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages'];import onnx
from deployment_candidates import load_deployment
from materialize_aliases import materialize
lease=Path('/data/zhangbenzhuang/huawei_sr/runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2)
base='cal_preserve_mean6_body3_native4_alignpixel';m=load_deployment(a.scene,base,r);k=load_deployment(a.scene,base+'_kernel7',r);q=load_deployment(a.scene,'quarter_w32_3x3_body1_gtweak',r);s=load_deployment(a.scene,'quarter_w32_3x3_body1_gtweak_refsplit',r)
probes=[('preserved_temporal_point',m.front.temporal_conv,(1,9,1024,1280)),('preserved_first6',m.front.first,(1,16,1024,1280)),('preserved_first7',k.front.first,(1,16,1024,1280)),('preserved_last3',m.front.last,(1,m.front.first.out_channels,512,640)),('gt_fused_projection',q.output.conv,(1,32,256,320)),('split_reference_embed',s.reference_embed,(1,12,64,64)),('split_reference_phase',s.reference_phase,(1,16,256,320)),('split_reference_resize16',nn.Upsample(size=(256,320),mode='bilinear',align_corners=False),(1,16,64,64)),('quarter_two2_deconv',load_deployment(a.scene,'quarter_w32_3x3_body1_gtweak_deconv',r).output,(1,32,256,320)),('quarter_two2_cascade',load_deployment(a.scene,'quarter_w32_3x3_body1_gtweak_cascade',r).output,(1,32,256,320))]
rows=[]
for i,(tag,layer,shape) in enumerate(probes):
 layer=layer.cuda().half().eval();x=torch.full(shape,.25,device='cuda',dtype=torch.float16);path=r/'probes'/f'{a.scene}_{tag}.onnx'
 if path.exists():raise FileExistsError(path)
 with torch.inference_mode():y=layer(x)
 torch.onnx.export(layer,x,str(path),input_names=['probe_input'],output_names=['probe_output'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(path));onnx.checker.check_model(g);onnx.save(g,path);rows.append({'file':path.name,'input_shape':list(shape),'input_dtype':'torch.float16','output_shape':list(y.shape),'runtime_input_seed':2900+i,'input_range':[-.25,.75],'full_model':False,'SDK_verified':False,'includes_independent_input_report_overhead':True});print('PROBE',tag,flush=True)
(r/'probes'/f'{a.scene}_finish_manifest.json').write_text(json.dumps({'scene':a.scene,'probes':rows},indent=2))
