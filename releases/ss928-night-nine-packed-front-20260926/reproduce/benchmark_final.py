"""Paired three-round timing of old and both student models, two explicit I/O routes."""
import fcntl,json,sys,gc
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr')
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/npu_graph_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
sys.path.insert(0,'/tmp/trajectory_student_20260926/runtime')
from load_student_model import load_student_model
from load_candidate import load_candidate
from graph_variants import FullGraphNine
from trajectory_student import StudentByteNine
from load_packed_model import load_packed_model
from output_variant import HalfShuffleStudent
out=root/'runs/SS928-PACKED-FRONT-20260926'
sys.path.insert(0,'/tmp/trajectory_student_20260926/runtime')
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
report={'GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'warmups':30,'rounds':3,'samples_per_round':200,'timing':'CUDA events, synchronous; preprocessing/transfers/compile excluded','TF32':False,'CUDA_graphs':False,'results':{}}
for scene,frame in [('ordinary',20),('special',60)]:
 cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:
  x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={}
 for route in ('fp32_gray','fp16_uint8'):
  report['results'][scene][route]={}
  for variant in ('base','compact8','s8_halfshuffle','joint12','joint8'):
   if variant=='base':
    core=load_candidate(cp,bp,remove_zero_init=True);m=FullGraphNine(core) if route=='fp32_gray' else core
   elif variant in ('joint12','joint8'):
    run=root/'runs'/('SS928-PACKED-FRONT-20260926' if variant=='joint12' else 'SS928-PACKED-FRONT-COMPACT8-20260926')
    step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step']
    m=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp,output='uint8' if route=='fp16_uint8' else 'float32')
   else:
    run=root/'runs'/('SS928-TRAJECTORY-STUDENT-20260926' if variant=='wide12' else 'SS928-TRAJECTORY-STUDENT-COMPACT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step']
    m=load_student_model(run/f'{scene}_student_{step:06d}.pt',cp,bp)
    if variant=='s8_halfshuffle':m=HalfShuffleStudent(m.core,m.student,output='uint8' if route=='fp16_uint8' else 'float32')
    elif route=='fp16_uint8':m=StudentByteNine(m.core,m.student)
   inp=x if route=='fp32_gray' else x.half();m=m.eval()
   # Reset prevents old graph reuse across experimental structures. Each variant
   # also has an explicit forward, and compiled-vs-source differences are saved.
   torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
   with torch.inference_mode():
    source=m(inp,ctx).float();value=exe(inp,ctx).float();d=(source-value).abs()
    numerical={'compiled_vs_source_max_gray':float(d.max()),'compiled_vs_source_mean_gray':float(d.mean())};del source,value,d
    for _ in range(30):exe(inp,ctx)
    torch.cuda.synchronize();rounds=[]
    for repeat in range(3):
     vals=[]
     for _ in range(200):
      a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();exe(inp,ctx);b.record();b.synchronize();vals.append(a.elapsed_time(b))
     rounds.append({'mean_ms':float(np.mean(vals)),'p95_ms':float(np.percentile(vals,95)),'samples_ms':vals})
   record={'mean_ms':float(np.mean([r['mean_ms'] for r in rounds])),'p95_ms':float(np.percentile([v for r in rounds for v in r['samples_ms']],95)),'rounds':rounds,'numerical':numerical}
   report['results'][scene][route][variant]=record;(out/'final_gpu_timing.json').write_text(json.dumps(report,indent=2))
   print(scene,route,variant,record['mean_ms'],record['p95_ms'],numerical,flush=True)
   del m,exe;gc.collect();torch.cuda.empty_cache()
 del x,ctx;torch.cuda.empty_cache()
print('FINISHED',flush=True)
