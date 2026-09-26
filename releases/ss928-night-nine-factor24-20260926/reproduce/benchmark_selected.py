"""Final paired timing of frozen current and selected conservative approximation."""
import fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-NIGHT-NINE-REFINE-20260926'
sys.path.insert(0,str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'));sys.path.insert(0,'/tmp')
from runtime_nine import load_model
from refine_variants import FactorizedFirst
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
(out/'selected_models').mkdir(exist_ok=False)
report={}
for scene,frame in [('ordinary',20),('special',60)]:
 modelpath=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/(scene+'_fused.pt')
 covariance=out/(scene+'_training_covariance.npy')
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as vec:
  x=torch.from_numpy(vec['nine_raw']).cuda().half();ctx=torch.from_numpy(vec['reference_thumb']).cuda()
 reference=None;report[scene]={}
 for variant in ('current_compiled','factor24_compiled','current_source','factor24_source'):
  m=load_model(modelpath,execution='source')
  if variant.startswith('factor24'):
   m.first=FactorizedFirst(m.first,24,np.load(covariance)).cuda()
   if variant.endswith('source'):
    weights={name:tensor.detach().cpu() for name,tensor in m.state_dict().items()}
    torch.save({'format':'ss928_nine_factorized_candidate_v1','scene':scene,'rank':24,'baseline_artifact_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'covariance_sha256':hashlib.sha256(covariance.read_bytes()).hexdigest(),'model':weights},out/'selected_models'/(scene+'_factor24.pt'))
  executable=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False}) if variant.endswith('compiled') else m
  with torch.inference_mode():
   y=executable(x,ctx);torch.cuda.synchronize()
   if reference is None:reference=y.clone()
   difference=(y.to(torch.int16)-reference.to(torch.int16)).abs()
   numerical={'max_gray':int(difference.max()),'mean_gray':float(difference.float().mean())}
   del y,difference
   for _ in range(30):executable(x,ctx)
   torch.cuda.synchronize();rounds=[]
   for repeat in range(3):
    samples=[]
    for _ in range(200):
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();executable(x,ctx);b.record();b.synchronize();samples.append(a.elapsed_time(b))
    rounds.append({'mean_ms':float(np.mean(samples)),'p95_ms':float(np.percentile(samples,95)),'samples_ms':samples})
  report[scene][variant]={'rounds':rounds,'numerical_vs_current_compiled':numerical}
  print(scene,variant,[v['mean_ms'] for v in rounds],numerical,flush=True)
  del executable,m;torch.cuda.empty_cache()
 del x,ctx,reference;torch.cuda.empty_cache()
(out/'selected_timing.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'protocol':'Fixed real 1024x1280 nine-frame float16 input and float32 64x64 context; complete 3072x3840 uint8 output; three rounds of 200 synchronized CUDA-event measurements, 30 warmups. Cold compilation, RAW preprocessing, H2D and D2H excluded. Source and compiled routes separated. No NPU timing evidence.','results':report},indent=2))
