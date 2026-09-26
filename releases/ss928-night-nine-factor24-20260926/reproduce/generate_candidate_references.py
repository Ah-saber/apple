"""Save source and compiler references for the exact frozen deployment candidate."""
import fcntl, json, sys
from pathlib import Path
import numpy as np
import torch

root=Path('/data/zhangbenzhuang/huawei_sr')
out=root/'runs/SS928-NIGHT-NINE-REFINE-20260926'
sys.path.insert(0,str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'))
sys.path.insert(0,'/tmp')
from load_candidate import load_candidate
lease=(root/'runs/TASK-019-gpu1.lock').open('a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
(out/'reference_outputs').mkdir(exist_ok=True)
report={}
for scene,frame in [('ordinary',20),('special',60)]:
    with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz',allow_pickle=False) as v:
        raw=v['nine_raw'].copy(); context=v['reference_thumb'].copy()
    arrays={}; report[scene]={}
    for precision in ('float16','float32'):
        x=torch.from_numpy(np.ascontiguousarray(raw,dtype=getattr(np,precision))).cuda()
        thumb=torch.from_numpy(context).cuda()
        model=load_candidate(out/'selected_models'/(scene+'_factor24.pt'),root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/(scene+'_fused.pt'),remove_zero_init=True)
        with torch.inference_mode():
            arrays[precision+'_source']=model(x,thumb).cpu().numpy()
            executable=torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
            arrays[precision+'_compiled']=executable(x,thumb).cpu().numpy()
        diff=np.abs(arrays[precision+'_source'].astype(np.int16)-arrays[precision+'_compiled'].astype(np.int16))
        report[scene][precision]={'shape':list(arrays[precision+'_source'].shape),'max_source_compiled_gray':int(diff.max()),'mean_source_compiled_gray':float(diff.mean())}
        print(scene,precision,report[scene][precision],flush=True)
        del model,executable,x,thumb;torch.cuda.empty_cache()
    np.savez_compressed(out/'reference_outputs'/(scene+'_reference_outputs.npz'),**arrays)
(out/'candidate_reference_verification.json').write_text(json.dumps(report,indent=2))
