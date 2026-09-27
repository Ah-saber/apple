import argparse,fcntl,gc,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tag',default='joint_benchmark');p.add_argument('--cases',nargs='+',default=['baseline','front_init','front_refined','up2','up3','up6','up2_f32','joint_refined_up2']);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-BODY-DISTILL-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from input_candidates import load_input as load_joint,prepare_inputs
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
frame=20 if a.scene=='ordinary' else 60;v=np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{a.scene}_frame_{frame}.npz');x=torch.from_numpy(v['nine_raw']).cuda();ctx=torch.from_numpy(v['reference_thumb']).cuda();cases=a.cases;report={'scene':a.scene,'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'input_shape':list(x.shape),'full_output_shape':[1,1,3072,3840],'TF32':False,'CUDA_graphs':False,'NPU_verified':False,'results':[]}
with torch.inference_mode():
 baseline=load_joint(a.scene,'baseline',run)(x,ctx).float()
 for case in cases:
  model=load_joint(a.scene,case,run);xx,cc=prepare_inputs(model,x,ctx);torch.compiler.reset();torch._inductor.config.triton.cudagraphs=False;exe=torch.compile(model,fullgraph=True);source=model(xx,cc);compiled=exe(xx,cc);d=(source-baseline).abs();cd=(compiled-source).abs();row={'case':case,'declared_output_dtype':str(source.dtype),'declared_input_shape':list(xx.shape),'declared_input_dtype':str(xx.dtype),'input_preparation_excluded':True,'source_mean_gray':float(d.mean()),'source_max_gray':float(d.max()),'compiled_own_source_mean_gray':float(cd.mean()),'compiled_own_source_max_gray':float(cd.max())}
  for _ in range(30):exe(xx,cc)
  samples=[]
  for _ in range(3):
   times=[]
   for _ in range(200):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();exe(xx,cc);end.record();end.synchronize();times.append(start.elapsed_time(end))
   samples.append(times)
  row.update(mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples);report['results'].append(row);(run/f'{a.scene}_{a.tag}.json').write_text(json.dumps(report,indent=2));print('BENCH',a.scene,case,{k:v for k,v in row.items() if k!='samples_ms'},flush=True)
  del exe,model,source,compiled;gc.collect();torch.compiler.reset();torch.cuda.empty_cache()
print('COMPLETE',a.scene,flush=True)
