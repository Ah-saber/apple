"""RAW-only full-sequence comparisons; fixed display scale, no webpages or local downloads."""
import argparse,copy,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
C=Path(__file__).resolve().parents[1];sys.path.insert(0,str(C/'src'))
from ir_sr.training import dataset_for_config,sha,atomic_json
from ir_sr.model import inference_model
from ir_sr.student_deployment import FullFrameStudent
p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
R=Path('/data/zhangbenzhuang/huawei_sr/runs')
if a.device=='cuda':
 assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1'
 lease=(R/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
entries={v['key']:v for v in json.loads(a.models.read_text())['models'] if v['key'] in ('s03_8k','s03_best','s03_last')}
assert set(entries)=={'s03_8k','s03_best','s03_last'}
models={};configs={}
for k,v in entries.items():
 assert sha(v['checkpoint'])==v['sha256'];state=torch.load(v['checkpoint'],map_location='cpu',weights_only=False);assert state['progress']['step']==v['step']
 configs[k]=state['config'];models[k]=FullFrameStudent(inference_model(state['config'],state['model']),hierarchical_pooling=True).to(a.device).eval();del state

def u8(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def panel(arrays,labels,size):
 w,h=size;out=Image.new('RGB',(w*len(arrays),h+32),'#202020');draw=ImageDraw.Draw(out)
 for i,(arr,label) in enumerate(zip(arrays,labels)):
  im=Image.fromarray(u8(arr));im=im.resize((w,h),Image.Resampling.LANCZOS);out.paste(im,(i*w,32));draw.text((i*w+8,8),label,fill='white')
 return np.asarray(out)
def writer(path,w,h):
 return subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','12','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-n',str(path)],stdin=subprocess.PIPE)
results=[];start=time.monotonic()
with torch.inference_mode():
 for split in ('val','test'):
  ds=dataset_for_config(configs['s03_8k'],split);root=Path(configs['s03_8k']['data_root']);selected={}
  for r in ds.records:selected.setdefault(r['scene_id'],r)
  for scene,r in selected.items():
   assert r['evaluation_scope']=='capture_group_development' and 'train_roi_tlhw' not in r
   raw=np.load(root/r['raw']['path'],mmap_mode='r');cal=ds.sequence_normalization.by_key[r['domain']+'/'+r['sequence_id']]
   assert raw.shape[1:]==(1024,1280) and len(raw)==cal['frames']
   n=min(2,len(raw)) if a.smoke else len(raw);folder=a.output/f'{split}_{scene}';folder.mkdir();fullpath=folder/'comparison_full_sequence.mp4';detailpath=folder/'comparison_x3_detail.mp4'
   labels=['RAW normalized','GT',*[entries[k]['label'] for k in models]]
   detail_labels=['RAW resized3x','GT resized3x reference',*[entries[k]['label'] for k in models]]
   fullwriter=writer(fullpath,640*len(labels),544);detailwriter=writer(detailpath,480*len(labels),272)
   records=[];previous={};boundary={s['begin'] for s in cal['segments'] if s['begin']};sources=[]
   try:
    for frame in range(n):
     gtpath=(root/r['target']['path']).with_name(f'{frame:06d}.png')
     with Image.open(gtpath) as im:assert im.mode=='L';gt=np.asarray(im,dtype=np.float32)/255.
     assert gt.shape==(1024,1280);sources.append(dict(frame=frame,path=str(gtpath),sha256=sha(gtpath)))
     norm=ds.normalization_for(r,frame);x=(raw[frame].astype(np.float32)-norm['offset'])/norm['scale'];tensor=torch.from_numpy(x[None,None]).to(a.device)
     views=[x,gt];details=[cv2.resize(x[384:464,480:640],(480,240),interpolation=cv2.INTER_LINEAR),cv2.resize(gt[384:464,480:640],(480,240),interpolation=cv2.INTER_LINEAR)];record=dict(frame=frame,segment_cut=frame in boundary,normalization=norm,gt_mean_gray=float(gt.mean()*255))
     for k,m in models.items():
      pred=m(tensor);assert torch.isfinite(pred).all();clipped=pred.clamp(0,1)
      small=torch.nn.functional.avg_pool2d(clipped,3)[0,0].cpu().numpy();detail=clipped[0,0,1152:1392,1440:1920].cpu().numpy();views.append(small);details.append(detail)
      residual=(small-gt)[3:-3,3:-3];v=dict(mean_gray=float(small.mean()*255),bias_gray=float(residual.mean()*255),std_gray=float(small.std()*255))
      if k in previous and frame not in boundary:v['consecutive_residual_change_mae_gray']=float(np.mean(np.abs(residual-previous[k]))*255)
      previous[k]=residual;record[k]=v
      del pred,clipped
     fullwriter.stdin.write(panel(views,labels,(640,512)).tobytes());detailwriter.stdin.write(panel(details,detail_labels,(480,240)).tobytes());records.append(record)
     if frame%20==0:print(split,scene,frame,'/',n,flush=True)
   finally:
    fullwriter.stdin.close();detailwriter.stdin.close()
    rc1=fullwriter.wait();rc2=detailwriter.wait()
   assert rc1==0 and rc2==0
   media=[]
   for path in (fullpath,detailpath):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True));assert int(probe['streams'][0]['nb_read_frames'])==n
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,timeout=180)
    media.append(dict(path=str(path),sha256=sha(path),**probe['streams'][0]))
   summary={k:dict(bias_mean_gray=float(np.mean([v[k]['bias_gray'] for v in records])),bias_std_over_time_gray=float(np.std([v[k]['bias_gray'] for v in records])),consecutive_residual_change_mae_gray=float(np.mean([v[k]['consecutive_residual_change_mae_gray'] for v in records if 'consecutive_residual_change_mae_gray' in v[k]])) if any('consecutive_residual_change_mae_gray' in v[k] for v in records) else None) for k in models}
   item=dict(split=split,scene=scene,sequence=r['sequence_id'],frames=n,source_raw=str(root/r['raw']['path']),source_raw_sha256=sha(root/r['raw']['path']),targets=sources,models=entries,summary=summary,records=records,media=media)
   atomic_json(folder/'report.json',item);results.append(item)
atomic_json(a.output/'report.json',dict(status='smoke_only' if a.smoke else 'complete',sequences=results,wall_seconds=time.monotonic()-start,protocol='All consecutive frames in four held-out capture sequences. Formal metrics remain on frozen val/test manifests. Playback12fps is a viewing choice, acquisition FPS unverified. Fixed0..1 display. 3x GT is resized style reference, not true HR. Temporal residual change uses consecutive (prediction-GT) differences, excluding frozen segment cuts; no motion alignment, so cannot uniquely measure flicker. Offline RAW normalization still uses full-sequence statistics.'))
