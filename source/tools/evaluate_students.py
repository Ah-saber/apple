"""Equal-budget quality audit and shared-GPU inference timing; no web pages."""
import argparse,csv,fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from ir_sr.training import dataset_for_config,atomic_json,sha
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
from ir_sr.metrics import image_metrics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args();OUT=a.output;OUT.mkdir(exist_ok=False,parents=True)
RUNS=Path('/data/zhangbenzhuang/huawei_sr/runs')
if a.device=='cuda':
 lease=(RUNS/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1'
 torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
spec=json.loads(a.models.read_text());entries={v['key']:v for v in spec['models']};assert len(entries)==len(spec['models']) and 'baseline' in entries
run_paths={k:Path(v['run']) for k,v in entries.items()}
models={};configs={};identities={};rows=[];parity=[];pictures=[];start=time.monotonic()
for k,r in run_paths.items():
 assert json.loads((r/'completed.json').read_text())['status']=='completed'
 ckpt=Path(entries[k]['checkpoint']);assert sha(ckpt)==entries[k]['sha256']
 state=torch.load(ckpt,map_location='cpu',weights_only=False);assert state['progress']['step']==entries[k]['step']
 configs[k]=state['config'];models[k]=inference_model(state['config'],state['model']).to(a.device).eval()
 identities[k]=dict(path=str(ckpt),sha256=sha(ckpt),training_commit=json.loads((r/'training_identity.json').read_text())['code_commit'],step=state['progress']['step'],label=entries[k]['label'])
 if (r/'continuation_parent.json').exists():
  lineage=json.loads((r/'continuation_parent.json').read_text())
  if state['progress']['step']<=lineage['parent_config']['max_steps']:
   identities[k]['inherited_from']=lineage['parent_run'];identities[k]['training_commit']=lineage['parent_training_identity']['code_commit']
 del state

def measure(pred,gt):
 assert torch.isfinite(pred).all()
 m=image_metrics(pred,gt,border=3);m.update(mean_bias=float((pred.clamp(0,1)-gt).mean()*255),mean_prediction=float(pred.clamp(0,1).mean()*255),mean_gt=float(gt.mean()*255),std_prediction=float(pred.clamp(0,1).std(unbiased=False)*255),std_gt=float(gt.std(unbiased=False)*255));return m

def panel(path,arrays,labels):
 ims=[]
 for arr in arrays:
  im=Image.fromarray(np.rint(arr.detach().float().cpu().numpy().squeeze().clip(0,1)*255).astype(np.uint8));im.thumbnail((640,520),Image.Resampling.LANCZOS);ims.append(im)
 w,h=ims[0].size;canvas=Image.new('RGB',(len(ims)*(w+8),h+32),'#202020');draw=ImageDraw.Draw(canvas)
 for i,(im,label) in enumerate(zip(ims,labels)):canvas.paste(im,(i*(w+8),28));draw.text((i*(w+8)+4,8),label,fill='white')
 canvas.save(path,quality=95);pictures.append(str(path))

with torch.inference_mode():
 for split in ['val','test']:
  ds={k:dataset_for_config(c,split) for k,c in configs.items()}
  for i,r in enumerate(ds['baseline'].records):
   if a.smoke and r['frame_id']!=0:continue
   items={k:d[i] for k,d in ds.items()};base=items['baseline'];gt=base['gt'][None].to(a.device);args={n:base[n][None].to(a.device) for n in ['context','context_box']};x=base['raw'][None].to(a.device)
   full=ds['baseline'].full_raw(i)[None].to(a.device);ctx,box=ds['baseline'].context_for(r);full_args=dict(context=ctx[None].to(a.device),context_box=box[None].to(a.device));crop=base['crop_tlhw'];t,l,h,w=crop
   row=dict(split=split,scene=r['scene_id'],sample_id=r['sample_id'],frame=r['frame_id'],crop_tlhw=list(crop));outputs={'reduced':{},'native':{}};outputs_x3={}
   for k,m in models.items():
    for field in ['raw','gt','context','context_box']:assert torch.equal(items[k][field],base[field]),(k,field)
    pred=m(x,**args);row[k+'_reduced']=measure(pred,gt);outputs['reduced'][k]=pred
    pfull=m(full,**full_args)
    ph=pfull.clamp(0,1)[...,12:-12,12:-12].reshape(1,1,254,12,318,12).mean((0,1,2,4))
    row[k+'_phase12_mean_range_gray']=float((ph.max()-ph.min())*255)
    row[k+'_phase12_mean_std_gray']=float(ph.std(unbiased=False)*255)
    if r['frame_id']==0:outputs_x3[k]=pfull[...,384*3:464*3,480*3:640*3].detach().cpu()
    row[k+'_native_full_out_of_range_fraction']=float(((pfull<0)|(pfull>1)).float().mean())
    outputs['native'][k]=torch.nn.functional.avg_pool2d(pfull.clamp(0,1),3)[...,t:t+h,l:l+w];row[k+'_native']=measure(outputs['native'][k],gt)
    if r['frame_id']==0:
     deploy=FullFrameStudent(m).to(a.device).eval();pdeploy=deploy(full);delta=(pdeploy-pfull).abs();error=float(delta.max());assert error<1e-4,(k,error)
     parity.append(dict(model=k,sample_id=r['sample_id'],max_abs=error,mae=float(delta.mean())))
     if True:
      fast=FullFrameStudent(m,hierarchical_pooling=True).to(a.device).eval();pf=fast(full);fast_error=float((pf-pdeploy).abs().max());assert fast_error<1e-4,fast_error
      parity.append(dict(model=k+'_fastpool',sample_id=r['sample_id'],max_abs=fast_error,mae=float((pf-pdeploy).abs().mean())))
      del fast,pf
     del deploy,pdeploy,delta
    del pfull
   if r['frame_id']==0:
    rawvis=torch.nn.functional.interpolate(base['raw'][None],size=gt.shape[-2:],mode='nearest')
    for mode,yy in outputs.items():panel(OUT/f'{split}_{r["scene_id"]}_{mode}.jpg',[rawvis,gt,*[yy[k] for k in models]],['Normalized RAW','GT',*[entries[k]['label'] for k in models]])
    panel(OUT/f'{split}_{r["scene_id"]}_native_detail.jpg',[arr[...,220:740,320:960] for arr in [gt,*[outputs['native'][k] for k in models]]],['GT',*[entries[k]['label'] for k in models]])
   if r['frame_id']==0:
    resized_gt=torch.nn.functional.interpolate(gt[...,384-t:464-t,480-l:640-l],scale_factor=3,mode='bilinear',align_corners=False)
    panel(OUT/f'{split}_{r["scene_id"]}_native_x3_detail.jpg',[resized_gt,*[outputs_x3[k] for k in models]],['GT resized3x (reference)',*[entries[k]['label'] for k in models]])
   rows.append(row);print(split,r['scene_id'],r['frame_id'],{k:round(row[k+'_native']['psnr'],3) for k in models},flush=True)
   del outputs,outputs_x3,gt,args,x,full,full_args,pred
summary={}
for sp in ['val','test']:
 summary[sp]={}
 for scene in configs['baseline']['scene_ids']:
  rr=[r for r in rows if r['split']==sp and r['scene']==scene];assert len(rr)==(1 if a.smoke else 12)
  summary[sp][scene]={k+'_'+mode:{metric:float(np.mean([r[k+'_'+mode][metric] for r in rr])) for metric in ['psnr','ssim','mean_bias','mean_prediction','mean_gt','std_prediction','std_gt']} for k in models for mode in ['reduced','native']}
# Where present, cross-check the trainer's independent frozen evaluation.
formal_checks=[]
if not a.smoke:
 for k,run in run_paths.items():
  for sp in ['val','test']:
   path=run/f"evaluation/step_{entries[k]['step']:09d}_{sp}/metrics.json"
   if not path.exists():
    formal_checks.append(dict(model=k,split=sp,status='newly_evaluated_no_historical_score'));continue
   formal=json.loads(path.read_text())
   for scene,v in formal['scene_metrics'].items():
    assert abs(v['psnr']-summary[sp][scene][k+'_reduced']['psnr'])<1e-5
   formal_checks.append(dict(model=k,split=sp,status='matched'))
report=dict(status='smoke_only' if a.smoke else 'complete',formal_checks=formal_checks,model_spec=spec,identities=identities,summary=summary,images=rows,static_deployment_parity=parity,pictures=pictures,wall_seconds=time.monotonic()-start,
 protocol={'reduced':'RAW manifest crop area3 -> network3x vs GT. FP32, clip prediction, border3, no uint8 rounding.',
 'native':'Full RAW -> network3x -> clip0..1 -> area3 -> same manifest crop vs GT. No true3x HR GT.',
 'comparison':'Frozen 8k references versus validation-selected full-run best and last200k. Same samples/crops/normalization; training budget differs. Best selected only by crop validation PSNR. Test reused for development; not blind.',
 'display':'Fixed0..1 to uint8 for all panels; no separate contrast stretching. native_x3_detail keeps actual3x output pixels, GT bilinear3x for style reference only. Phase12 mean spread is diagnostic and affected by scene structure, not a quality acceptance metric.'})
atomic_json(OUT/'report.json',report)
with (OUT/'summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['split','scene','variant_mode','psnr','ssim','mean_bias'])
 for sp,scenes in summary.items():
  for scene,variants in scenes.items():
   for v,metrics in variants.items():w.writerow([sp,scene,v,*[metrics[q] for q in ['psnr','ssim','mean_bias']]])
print(json.dumps(summary),flush=True)
