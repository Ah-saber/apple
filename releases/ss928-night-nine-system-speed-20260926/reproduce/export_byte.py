import fcntl,sys,gc
from pathlib import Path
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926/source_onnx';base=Path(__file__).parent;sys.path[:0]=[str(base/'runtime'),str(base/'tools'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages');import onnx
from materialize_aliases import materialize
from build_system import load_case
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
for scene in ('ordinary','special'):
 model,_=load_case(root,scene,'balanced_low_rows16_nhwc16_byte')
 for small in (False,):
  x=torch.rand(1,1024,1280,9,dtype=torch.float16,device='cuda');c=torch.rand(1,1,64,64,device='cuda');path=out/f'{scene}_balanced_low_rows16_nhwc16_byte{"_small" if small else ""}.onnx';torch.onnx.export(model,(x,c),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);g,_=materialize(onnx.load(str(path)));onnx.save(g,str(path));print('EXPORT',path.name,flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
