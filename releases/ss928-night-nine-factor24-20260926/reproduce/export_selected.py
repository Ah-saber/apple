"""Export saved candidate graph for future NPU compilation; checker is not an NPU support test."""
import argparse,fcntl,json,sys
from collections import Counter
from pathlib import Path
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-NIGHT-NINE-REFINE-20260926'
sys.path.insert(0,str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'));sys.path.insert(0,'/tmp')
sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from load_candidate import load_candidate
p=argparse.ArgumentParser();p.add_argument('--scene',choices=('ordinary','special'),required=True);p.add_argument('--size',type=int,default=1024);p.add_argument('--width',type=int,default=1280);p.add_argument('--input-precision',choices=('float16','float32'),default='float16');p.add_argument('--remove-zero-init',action='store_true');a=p.parse_args()
torch.set_num_threads(2)
if a.size>=1024:
 lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
 device='cuda'
else:device='cpu'
m=load_candidate(out/'selected_models'/(a.scene+'_factor24.pt'),root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/(a.scene+'_fused.pt'),device=device,remove_zero_init=a.remove_zero_init)
x=torch.rand(1,9,a.size,a.width,dtype=torch.float16 if a.input_precision=='float16' else torch.float32,device=device);ctx=torch.rand(1,1,64,64,device=device)
path=out/'selected_models'/(a.scene+'_factor24_'+str(a.size)+'x'+str(a.width)+'_'+a.input_precision+('_nozero' if a.remove_zero_init else '')+'.onnx')
with torch.inference_mode():torch.onnx.export(m,(x,ctx),path,opset_version=17,dynamo=False,do_constant_folding=True,input_names=['nine_raw','reference_thumb'],output_names=['display_uint8'])
g=onnx.load(path);onnx.checker.check_model(g)
report={'scene':a.scene,'model':str(path),'shape':[1,9,a.size,a.width],'input_types':[i.type.tensor_type.elem_type for i in g.graph.input],'output_type':g.graph.output[0].type.tensor_type.elem_type,'static_check':True,'ops':dict(Counter(n.op_type for n in g.graph.node)),'npu_compilation_verified':False}
path.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
