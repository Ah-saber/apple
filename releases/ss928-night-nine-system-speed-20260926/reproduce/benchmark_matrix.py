import fcntl,gc,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from build_system import load_system
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
variants=['base','quarter','reference','body','quarter_ref','low_scale','quarter_ref_low','all_body_low','direct_base','direct_all','phase','all_phase','all_half_io','pre_byte','all_byte','factor8','factor12','all_low_rank8'];report={'variants':variants,'warmup':30,'rounds':3,'samples_per_round':200,'TF32':False,'CUDA_graphs':False,'GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'NPU_verified':False,'results':{}}
for scene,frame in [('ordinary',20),('special',60)]:
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={}
 for variant in variants:
  m=load_system(root,scene,variant);xx=x.half() if variant=='all_half_io' else x;cc=ctx.half() if variant=='all_half_io' else ctx;torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
  with torch.inference_mode():
   source=m(xx,cc).float();compiled=exe(xx,cc).float()
   if variant=='base':reference=source.cpu().numpy()
   delta=np.abs(source.cpu().numpy()-reference);record={'input_dtype':str(xx.dtype),'output_dtype':m.output_dtype,'source_vs_base_mae_gray':float(delta.mean()),'source_vs_base_max_gray':float(delta.max()),'compiled_vs_source_mae_gray':float((compiled-source).abs().mean()),'compiled_vs_source_max_gray':float((compiled-source).abs().max())};del source,compiled
   if variant.startswith('factor') or 'rank' in variant:record['retained_output_weight_energy']=m.core.model.upsample[0].retained_energy
   for _ in range(30):exe(xx,cc)
   torch.cuda.synchronize();rounds=[]
   for _ in range(3):
    values=[]
    for _ in range(200):
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();exe(xx,cc);e.record();e.synchronize();values.append(s.elapsed_time(e))
    rounds.append(values)
   flat=sum(rounds,[]);record.update(mean_ms=float(np.mean(flat)),p95_ms=float(np.percentile(flat,95)),samples_ms=rounds);report['results'][scene][variant]=record;print('RESULT',scene,variant,{k:v for k,v in record.items() if k!='samples_ms'},flush=True)
  del exe,m,xx,cc;gc.collect();torch.cuda.empty_cache();(out/'matrix_benchmark.json').write_text(json.dumps(report,indent=2))
 del x,ctx
print('FINISHED',flush=True)
