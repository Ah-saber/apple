"""Verify source model loading entirely from a separately assembled bundle."""
import json,sys,shutil,hashlib,fcntl
from pathlib import Path
import numpy as np
import torch,cv2
r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');b=r/'bundle_verification';b.mkdir(exist_ok=True);shutil.copytree(r/'code_snapshot/runtime',b/'runtime',dirs_exist_ok=True)
old=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BODY-DISTILL-20260927');(b/'body_models').mkdir(exist_ok=True)
for scene in ['ordinary','special']:
 for d in [2,3]:shutil.copy2(old/f'{scene}_body{d}_quantsearch.pt',b/'body_models')
(b/'body_models/frozen_models').mkdir(exist_ok=True)
for f in Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927').glob('*.pt'):
 if 'output_quantsearch' in f.name or 'reference_relu_output_stable' in f.name:shutil.copy2(f,b/'body_models/frozen_models'/f.name)
for name in ['ss928-night-nine-system-speed-20260926','ss928-night-nine-packed-front-20260926']:
 src=Path('/tmp/ss928_release_verification_20260926')/name
 for child in ['runtime','models']:shutil.copytree(src/child,b/'frozen_system_dependencies'/name/child,dirs_exist_ok=True)
for f in r.glob('*.pt'):shutil.copy2(f,b/f.name)
sys.path.insert(0,str(b/'runtime'));from deployment_candidates import load_deployment,prepare_inputs
lease=(r.parent/'TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;results=[]
with torch.inference_mode():
 for scene in ['ordinary','special']:
  frame=20 if scene=='ordinary' else 60;v=np.load(r/'test_vectors'/f'{scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda()
  for case in ['body3_quantsearch','cal_preserve_mean6_body3_native4_alignpixel','cal_quarter_w32_3x3_body1_gtweak']:
   m=load_deployment(scene,case,b);xx,cc=prepare_inputs(m,x,c);y=m(xx,cc).cpu().numpy();expected=np.load(r/'reference_outputs'/f'{scene}_{case}.npz')['output'];err=np.abs(y.astype(np.float32)-expected.astype(np.float32));assert float(err.mean())<=.05;assert float(err.max())<=1;results.append({'scene':scene,'case':case,'shape':list(y.shape),'mean_gray':float(err.mean()),'max_gray':float(err.max()),'bundled_dependencies':True})
(r/'bundle_verification.json').write_text(json.dumps({'results':results,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'SDK_verified':False},indent=2));print('BUNDLE_VERIFIED')
