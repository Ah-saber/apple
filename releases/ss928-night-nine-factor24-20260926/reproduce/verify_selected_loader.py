import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-NIGHT-NINE-REFINE-20260926'
sys.path.insert(0,str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'));sys.path.insert(0,'/tmp')
from runtime_nine import load_model
from refine_variants import FactorizedFirst
from load_candidate import load_candidate
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
result={}
for scene,frame in [('ordinary',20),('special',60)]:
 baseline=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/(scene+'_fused.pt')
 expected=load_model(baseline,device='cpu');expected.first=FactorizedFirst(expected.first,24,np.load(out/(scene+'_training_covariance.npy')))
 actual=load_candidate(out/'selected_models'/(scene+'_factor24.pt'),baseline,device='cpu')
 assert expected.state_dict().keys()==actual.state_dict().keys()
 assert all(torch.equal(t,actual.state_dict()[k]) for k,t in expected.state_dict().items())
 expected.cuda();actual.cuda()
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:
  x=torch.from_numpy(v['nine_raw']).cuda().half();context=torch.from_numpy(v['reference_thumb']).cuda()
 with torch.inference_mode():
  one=expected(x,context);two=actual(x,context);assert torch.equal(one,two)
 result[scene]={'exact_weights':True,'full_source_output_identical':True,'output_shape':list(two.shape),'dtype':str(two.dtype)}
 print(scene,result[scene],flush=True)
 del expected,actual,one,two,x,context;torch.cuda.empty_cache()
(out/'selected_loader_verification.json').write_text(json.dumps(result,indent=2))
