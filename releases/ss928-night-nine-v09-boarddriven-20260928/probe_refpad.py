"""Real full-frame source checks before expensive sequence evaluation."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
run=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V09-20260928')
sys.path.insert(0,str(Path(__file__).parent))
from board_candidates import load_board,prepare_inputs
lease=(run.parent/'TASK-019-gpu1.lock').open('a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cases=['v08_c11','rows32','rows32_meanfirst','rows32_refpad16','rows32_refpad16_meanfirst']
rows=[]
with torch.inference_mode():
 for scene,frame in [('ordinary',20),('special',60)]:
  v=np.load(run/'v08/test_vectors'/f'{scene}_frame_{frame}.npz')
  x=torch.from_numpy(v['nine_raw']).cuda();c=torch.from_numpy(v['reference_thumb']).cuda()
  ref=None
  for case in cases:
   model=load_board(scene,case,run)
   xx,cc=prepare_inputs(model,x,c)
   y=model(xx,cc).float()
   if ref is None:ref=y
   d=(y-ref).abs()
   item={'scene':scene,'frame':frame,'case':case,'input_shape':list(xx.shape),'output_shape':list(y.shape),'mean_gray_difference':float(d.mean()),'max_gray_difference':float(d.max()),'same_pixels':float((d==0).float().mean()),'NPU_verified':False}
   rows.append(item)
   print(item,flush=True)
(run/'refpad_probe.json').write_text(json.dumps(rows,indent=2))
