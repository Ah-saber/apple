"""Small bounded GPU preflight; never reuse probe weights for training."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from ir_sr.model import inference_model,use_shuffle23,to_deploy
from ir_sr.student_deployment import FullFrameStudent
from ir_sr.auxiliary import training_model,loss_terms
from ir_sr.training import dataset_for_config,epoch_loader,seed_all,atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
code=Path(__file__).resolve().parents[1];runs=Path('/data/zhangbenzhuang/huawei_sr/runs');base=runs/'GLOBAL-REFERENCE-V1-20260923-LIGHT_MEDIUM-8K'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True;torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
start=time.monotonic();cfg=json.loads((base/'config.json').read_text());ckpt=base/'checkpoints/step_000008000.pt';s=torch.load(ckpt,map_location='cpu',weights_only=False)
m=inference_model(cfg,s['model']).cuda().eval();split=use_shuffle23(copy.deepcopy(m));static=FullFrameStudent(split).cuda().eval()
ds=dataset_for_config(cfg,'val');rows=[]
with torch.no_grad():
 for scene in cfg['scene_ids']:
  i=next(i for i,r in enumerate(ds.records) if r['scene_id']==scene);r=ds.records[i]
  x=ds.full_raw(i)[None].cuda();context,box=ds.context_for(r);args=dict(context=context[None].cuda(),context_box=box[None].cuda())
  y=m(x,**args);z=split(x,**args);w=static(x)
  phase=float((y-z).abs().max());deploy=float((y-w).abs().max())
  thumb=float((torch.nn.functional.avg_pool2d(x,(16,20))-args['context']).abs().max())
  assert phase<1e-4 and deploy<1e-4,(phase,deploy)
  rows.append(dict(scene=scene,sample_id=r['sample_id'],shuffle_max_abs=phase,static_max_abs=deploy,thumbnail_max_abs=thumb))
  del x,y,z,w,args
 del m,split,static,s
torch.cuda.empty_cache();probes=[]
for tag in ['s01','s02']:
 c=json.loads((code/f'configs/train/student_{tag}_light_medium_8k.json').read_text());seed_all(c['seed'])
 ds=dataset_for_config(c,'train',use_cache=True);loader,_=epoch_loader(ds,c,0);iterator=iter(loader)
 m=training_model(c).cuda();opt=torch.optim.Adam(m.parameters(),lr=c['lr'],betas=(.9,.99));losses=[]
 torch.cuda.reset_peak_memory_stats()
 for step in range(3):
  b=next(iterator);v={k:b[k].cuda() for k in ['raw','gt','middle_raw','raw_loss_multiplier','context','context_box']};opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   loss,display,aux=loss_terms(m,v['raw'],v['gt'],v['middle_raw'],.1,v['raw_loss_multiplier'],v['context'],v['context_box'])
  assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
  losses.append(dict(total=float(loss),display=float(display),auxiliary=float(aux),gradient=float(grad)))
 assert m.global_reference.project.weight.abs().sum()>0
 probes.append(dict(variant=tag,batch_size=c['batch_size'],losses=losses,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
 del m,opt,v,b,iterator,loader,ds,loss,display,aux;torch.cuda.empty_cache()
report=dict(status='passed',parent_checkpoint=str(ckpt),parent_sha256=sha(ckpt),full_frame_checks=rows,training_probes=probes,seconds=time.monotonic()-start,weights_reused=False)
atomic_json(a.output,report);print(json.dumps(report),flush=True)
