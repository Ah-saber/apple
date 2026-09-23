"""Untrained architecture latency screening; no quality claims for random weights."""
import argparse,copy,csv,fcntl,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,onnx,onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from ir_sr.auxiliary import training_model,loss_terms
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
from ir_sr.training import atomic_json,dataset_for_config,epoch_loader,seed_all
from finalize_training import ExportGrayUnshuffle
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
C=Path(__file__).resolve().parents[1];R=Path('/data/zhangbenzhuang/huawei_sr/runs');lease=(R/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base=R/'SS928-STUDENT-V1-20260923-S02-LIGHT_MEDIUM-8K';c0=json.loads((base/'config.json').read_text());configs={'s02':c0}
for tag in ['s06']:configs[tag]=json.loads((C/f'configs/train/student_{tag}_light_medium_8k.json').read_text())
ds=dataset_for_config(c0,'val');raw=ds.full_raw(0)[None].cuda();ctx,box=ds.context_for(ds.records[0]);context=ctx[None].cuda();box=box[None].cuda();summary={};times=[];before=subprocess.check_output(['nvidia-smi'],text=True)
for tag,c in configs.items():
 seed_all(928)
 if tag=='s02':
  state=torch.load(base/'checkpoints/step_000008000.pt',map_location='cpu',weights_only=False);core=inference_model(c,state['model']);del state
 else:
  core=training_model(c)
  # Nonzero reference projection for a meaningful parity probe, not reusable weights.
  with torch.no_grad():
   core.global_reference.project.weight.normal_(std=.02)
   if c.get('raw_skip'):core.upsample[0].weight.normal_(std=.01)
 core=core.cuda().eval();model=FullFrameStudent(core).cuda().eval()
 macs=[];handles=[]
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.Conv2d):
   handles.append(module.register_forward_hook(lambda m,args,out,n=name:macs.append(dict(name=n,macs=int(out.numel()*m.weight[0].numel())))))
 with torch.inference_mode():
  gold=core(raw,context=context,context_box=box);pred=model(raw);error=float((gold-pred).abs().max());assert error<1e-4,(tag,error)
  for h in handles:h.remove()
  for _ in range(50):pred=model(raw)
  torch.cuda.synchronize();tt=[]
  for repeat in range(3):
   for i in range(200):
    e=torch.cuda.Event(enable_timing=True);f=torch.cuda.Event(enable_timing=True);e.record();pred=model(raw);f.record();f.synchronize();ms=e.elapsed_time(f);tt.append(ms);times.append(dict(model=tag,repeat=repeat,index=i,ms=ms))
 record=dict(trained=tag=='s02',mean_ms=float(np.mean(tt)),p50_ms=float(np.median(tt)),p95_ms=float(np.percentile(tt,95)),max_ms=float(np.max(tt)),conv_macs=sum(v['macs'] for v in macs),parameters=sum(p.numel() for p in model.parameters()),static_parity_max_abs=error)
 # CPU full-size ONNX check before investing in training.
 model=model.cpu();del core,gold,pred;torch.cuda.empty_cache();model.model.down=ExportGrayUnshuffle(c.get('packing_factor',2));out=a.output/tag;out.mkdir();dest=out/'UNTRAINED_probe.onnx' if tag!='s02' else out/'s02_reference.onnx'
 torch.onnx.export(model,torch.zeros(1,1,1024,1280),str(dest),input_names=['raw'],output_names=['display'],opset_version=17,dynamo=False)
 graph=SymbolicShapeInference.infer_shapes(onnx.load(dest),auto_merge=True);onnx.checker.check_model(graph,full_check=True);onnx.save(graph,dest)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;session=ort.InferenceSession(str(dest),opts,providers=['CPUExecutionProvider']);x=raw.cpu().numpy()
 with torch.inference_mode():y=model(torch.from_numpy(x)).numpy()
 z=session.run(None,{'raw':x})[0];diff=np.abs(z-y);assert np.isfinite(z).all() and float(diff.max())<1e-3
 record['onnx_max_abs']=float(diff.max());record['SS928']='untested; new SpaceToDepth4 / DepthToSpace2->2->3 parameters require board probe'
 del session,model,y,z,diff;torch.cuda.empty_cache()
 # Three real BF16 training updates with unchanged batch16 and all targets.
 if tag!='s02':
  seed_all(928);m=training_model(c).cuda();opt=torch.optim.Adam(m.parameters(),lr=c['lr'],betas=(.9,.99));dataset=dataset_for_config(c,'train',use_cache=True);loader,_=epoch_loader(dataset,c,0);iterator=iter(loader);losses=[];torch.cuda.reset_peak_memory_stats()
  for i in range(3):
   batch=next(iterator);v={k:batch[k].cuda() for k in ['raw','gt','middle_raw','raw_loss_multiplier','context','context_box']};opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,display,aux=loss_terms(m,v['raw'],v['gt'],v['middle_raw'],.1,v['raw_loss_multiplier'],v['context'],v['context_box'])
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();losses.append(float(loss))
  record['training_probe_losses']=losses;record['peak_train_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
  del m,opt,v,batch,iterator,loader,dataset,loss,display,aux;torch.cuda.empty_cache()
 summary[tag]=record;print(tag,json.dumps(record),flush=True)
with (a.output/'per_frame.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(times[0]));writer.writeheader();writer.writerows(times)
atomic_json(a.output/'report.json',dict(status='passed',models=summary,gpu_before=before,gpu_after=subprocess.check_output(['nvidia-smi'],text=True),protocol='Shared5090 FP32/TF32off, full1024x1280->3072x3840 including reference.50warmup+3x200 CUDA timing. S06 random probe only; no quality claims. Old S02 retimed in same session. No data preprocessing or CPU I/O time.'))
