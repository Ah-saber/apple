"""Full-sequence GT and temporal comparison of ties-even and AddHalf/Cast."""
import argparse,fcntl,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from front_candidates import load_front_candidate,prepare_inputs

def blocks(x):
    mids=[];spreads=[]
    for frame in x:
        h,w=frame.shape;middle=[];spread=[]
        for iy in range(8):
            for ix in range(8):
                area=frame[iy*h//8:(iy+1)*h//8,ix*w//8:(ix+1)*w//8]
                q=np.percentile(area,(10,50,90));middle.append(q[1]);spread.append(q[2]-q[0])
        mids.append(middle);spreads.append(spread)
    return np.asarray(mids),np.asarray(spreads)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['ordinary','special'],required=True);p.add_argument('--case',required=True);p.add_argument('--v09-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/data/zhangbenzhuang/huawei_sr');special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';training='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
    sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
    from ir_sr.training import dataset_for_config
    state=torch.load(root/'runs'/training/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene)
    ry,rx,rh,rw=rec['eval_crop_tlhw'];roi=(slice(ry,ry+rh),slice(rx,rx+rw));count=120 if special else 60
    with (a.v09_run.parent/'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_front_candidate(a.scene,a.case,a.v09_run).eval()
        series={'even':[],'addhalf_cast':[],'gt':[]};psnr={'even':[],'addhalf_cast':[]};changed=[];bias=[]
        with torch.inference_mode():
            for frame in range(count):
                row=dict(rec,frame_id=frame);x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();xx,cc=prepare_inputs(model,x,ctx)
                gray=model(xx,cc).half();even=gray.round().clamp(0,255);cast=torch.trunc((gray+torch.tensor(.5,device='cuda',dtype=torch.float16)).float()).clamp(0,255)
                delta=cast-even;changed.append(float((delta!=0).float().mean()));bias.append(float(delta.float().mean()))
                target=(Path(state['config']['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png');gt=np.asarray(Image.open(target),dtype=np.float32)[roi]
                series['gt'].append(gt[::2,::2].astype(np.float16))
                for name,value in (('even',even),('addhalf_cast',cast)):
                    pred=F.avg_pool2d(value.float(),3,3)[0,0].cpu().numpy()[roi]
                    error=(pred[3:-3,3:-3].astype(np.float64)-gt[3:-3,3:-3])/255
                    psnr[name].append(-10*math.log10(max(np.mean(error**2),1e-12)))
                    series[name].append(pred[::2,::2].astype(np.float16))
                if frame%10==0:print('FRAME',a.scene,frame,flush=True)
    arrays={k:np.stack(v).astype(np.float32) for k,v in series.items()};gt=arrays['gt'];mean=gt.mean(0);temporal=gt.std(0);import cv2
    grad=np.hypot(cv2.Sobel(mean,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(mean,cv2.CV_32F,0,1,ksize=3)/8);static=(temporal<=1)&(grad<=5)&(mean>5)&(mean<250);static[:4]=static[-4:]=False;static[:,:4]=static[:,-4:]=False;weak=(temporal>2)&(temporal<=8)&(mean>5)&(mean<250);gm,gs=blocks(gt)
    metrics={}
    for name in ('even','addhalf_cast'):
        prediction=arrays[name];error=prediction-gt;pm,ps=blocks(prediction)
        metrics[name]={'mean_psnr_db':float(np.mean(psnr[name])),'static_error_temporal_std_gray':float(error.std(0)[static].mean()),'weak_mae_gray':float(np.abs(error[:,weak]).mean()),'brightness_error_temporal_std_gray':float(np.median((pm-gm).std(0))),'contrast_error_temporal_std_gray':float(np.median((ps-gs).std(0)))}
    report={'scene':a.scene,'case':a.case,'frames':count,'roi':rec['eval_crop_tlhw'],'changed_pixel_fraction_mean':float(np.mean(changed)),'mean_gray_bias':float(np.mean(bias)),'metrics':metrics,'NPU_verified':False,'test_used_for_architecture_assessment':True}
    a.output.write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()
