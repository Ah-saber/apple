import fcntl,gc,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926';out.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from load_packed_model import load_packed_model
from system_variants import FoldedFront,QuarterFront,TransposeOutput,FullSystem
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
report={'GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,'warmup':30,'rounds':3,'samples_per_round':200,'TF32':False,'CUDA_graphs':False,'NPU_verified':False,'results':{}}
# Independent FP32 algebra check for phase-preserving output reparameterization.
conv=torch.nn.Conv2d(16,36,3,padding=1).cuda();random=torch.randn(1,16,16,24,device='cuda');expected=F.pixel_shuffle(conv(random),6)
for factor in [6,2]:
 actual=TransposeOutput(conv,factor)(random);delta=(actual-expected).abs();assert delta.max()<.00002;print('ALGEBRA',factor,float(delta.max()),flush=True)
del conv,random,actual,expected
for scene,frame in [('ordinary',20),('special',60)]:
 run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt'
 with np.load(root/'runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS'/f'{scene}_frame_{frame}.npz') as v:x=torch.from_numpy(np.ascontiguousarray(v['nine_raw'],dtype=np.float32)).cuda();ctx=torch.from_numpy(np.ascontiguousarray(v['reference_thumb'],dtype=np.float32)).cuda()
 report['results'][scene]={};source_base=None
 for name in ['base','fold32','fold16','transpose6','transpose2','quarter_untrained','half_input','half_output','byte_output']:
  base=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',cp,bp);front=FoldedFront(base.front,'float32' if name=='fold32' else 'float16') if name.startswith('fold') else (QuarterFront(8 if scene=='ordinary' else 12,2).cuda().half().to(memory_format=torch.channels_last) if name=='quarter_untrained' else None)
  m=FullSystem(base,front=front,output_mode=name if name.startswith('transpose') else 'shuffle',output_dtype='float16' if name=='half_output' else ('uint8' if name=='byte_output' else 'float32')).eval();xx=x.half() if name=='half_input' else x;cc=ctx.half() if name=='half_input' else ctx
  torch.compiler.reset();exe=torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False})
  with torch.inference_mode():
   source=m(xx,cc).float();compiled=exe(xx,cc).float()
   if name=='base':source_base=source.cpu().numpy()
   delta=np.abs(source.cpu().numpy()-source_base);num={'source_vs_base_mae_gray':float(delta.mean()),'source_vs_base_max_gray':float(delta.max()),'compiled_vs_source_mae_gray':float((compiled-source).abs().mean()),'compiled_vs_source_max_gray':float((compiled-source).abs().max())};del source,compiled
   for _ in range(30):exe(xx,cc)
   torch.cuda.synchronize();rounds=[]
   for _ in range(3):
    values=[]
    for _ in range(200):
     s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();exe(xx,cc);e.record();e.synchronize();values.append(s.elapsed_time(e))
    rounds.append(values)
   flat=sum(rounds,[]);r={'mean_ms':float(np.mean(flat)),'p95_ms':float(np.percentile(flat,95)),'samples_ms':rounds,'numerical':num,'input_dtype':str(xx.dtype),'output_dtype':m.output_dtype,'trained':name!='quarter_untrained'};report['results'][scene][name]=r;print('RESULT',scene,name,{k:v for k,v in r.items() if k!='samples_ms'},flush=True)
  del exe,m,front,base,xx,cc;gc.collect();torch.cuda.empty_cache();(out/'initial_benchmark.json').write_text(json.dumps(report,indent=2))
 del x,ctx
print('FINISHED',flush=True)
