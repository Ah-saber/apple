"""Fixed-weight history intervention; GT-only small-target and trail evaluation."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, gaussian_filter, label, find_objects

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup, GROUPS, sha


def foreground(sequence, road):
    background=np.median(sequence,axis=0)
    signal=sequence-background
    # Only a scalar drift estimated from road pixels; do not fit target amplitudes.
    drift=np.median(signal[:,road],axis=1)
    return signal-drift[:,None,None],background,drift


def small_targets(signal, road):
    masks=[];boxes=[]
    for frame in signal:
        mask=(gaussian_filter(frame,.7)>10)&road
        ids,count=label(mask)
        kept=np.zeros_like(mask);b=[]
        for i,slices in enumerate(find_objects(ids),1):
            if slices is None:continue
            yy,xx=slices;area=int((ids[slices]==i).sum())
            if 4<=area<=600 and xx.stop-xx.start<=60 and yy.stop-yy.start<=35:
                kept[slices]|=ids[slices]==i
                b.append([yy.start,xx.start,yy.stop-yy.start,xx.stop-xx.start,area])
        masks.append(kept);boxes.append(b)
    return np.stack(masks),boxes


def evaluate_signals(values, directory):
    names=list(values);gt=values['GT']
    height,width=gt.shape[-2:];road=np.zeros((height,width),np.uint8)
    # Polygon inspected on the GT road; coordinates are native RAW pixels minus y=240.
    polygon=np.array([[0,255-240],[1279,485-240],[1279,570-240],[0,340-240]])
    cv2.fillPoly(road,[polygon],1);road=road.astype(bool)
    signals={};backgrounds={};drifts={}
    for name,seq in values.items():signals[name],backgrounds[name],drifts[name]=foreground(seq,road)
    masks,boxes=small_targets(signals['GT'],road)
    per_frame=[]
    for t in range(8,len(gt)):
        current=masks[t]
        old=np.any(masks[t-8:t],axis=0)
        old=binary_dilation(old,iterations=1)&~binary_dilation(current,iterations=3)&road
        item={'frame':t,'current_pixels':int(current.sum()),'historical_only_pixels':int(old.sum()),'boxes_y_relative_to_240':boxes[t],'methods':{}}
        for name in names[1:]:
            pred=signals[name][t];truth=signals['GT'][t]
            record={}
            if current.any():
                record['current_target_positive_contrast_ratio']=float(np.maximum(pred[current],0).sum()/np.maximum(truth[current],0).sum())
                record['current_target_signal_mae_gray']=float(np.abs(pred[current]-truth[current]).mean())
            if old.any():
                record['historical_only_positive_excess_gray']=float(np.maximum(pred[old]-truth[old],0).mean())
                record['historical_only_signal_mae_gray']=float(np.abs(pred[old]-truth[old]).mean())
            item['methods'][name]=record
        per_frame.append(item)
    summaries={}
    for name in names[1:]:
        fields=set(k for row in per_frame for k in row['methods'][name])
        summaries[name]={k:float(np.mean([row['methods'][name][k] for row in per_frame if k in row['methods'][name]])) for k in fields}
    report={'mask_source':'GT only; positive contrast >10 after sigma .7; 4..600 pixels; box<=60x35; native road polygon',
            'ROI_yxhw':[240,0,height,width],'background':'offline 120-frame temporal median, each method separately; only scalar road drift removed',
            'motion_masks_used_in_training':False,'not_an_inference_operation':True,
            'summary':summaries,'per_frame':per_frame,'target_counts':[len(b) for b in boxes]}
    (directory/'small_target_history.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(directory/'road_signals_and_masks.npz',**signals,GT_current_mask=masks)
    # Fixed locations/frames for all methods, selected from GT, with nearest-neighbor display.
    ranked=sorted(range(8,len(gt)-4),key=lambda t:float(np.maximum(signals['GT'][t][masks[t]],0).sum()),reverse=True)
    chosen=[]
    for t in ranked:
        if boxes[t] and all(abs(t-v)>12 for v in chosen):chosen.append(t)
        if len(chosen)==3:break
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',16)
    for central in chosen:
        box=max(boxes[central],key=lambda b:b[4]);yy,xx,hh,ww,_=box
        top=max(0,min(height-80,yy+hh//2-40));left=max(0,min(width-128,xx+ww//2-64))
        canvas=Image.new('RGB',(len(names)*384,8*268),'#161b22');draw=ImageDraw.Draw(canvas)
        for row,t in enumerate(range(central-3,central+5)):
            for col,name in enumerate(names):
                image=Image.fromarray(np.rint(values[name][t,top:top+80,left:left+128].clip(0,255)).astype(np.uint8))
                canvas.paste(image.resize((384,240),Image.Resampling.NEAREST),(col*384,row*268+28))
                draw.text((col*384+4,row*268+3),f'{name} | {t:03d}',font=font,fill='white')
        canvas.save(directory/f'target_{central:03d}_eight_frames.png')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--groups',nargs='+',choices=['light','heavy'],default=['light','heavy']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    manifest={'NPU_measured':False,'weights_changed':False,'preprocessing_and_reference_unchanged':True,'scenes':{}}
    for group in a.groups:
        ck=torch.load(PREVIOUS/(group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)
        GROUPS[group].update(ck['spec'])
        compact,base,config,teacher,base_path,df,box_for=setup(group)
        compact.load_state_dict(ck['model']);compact=compact.cuda().float().eval();base=base.cuda().float().eval()
        data=df(config,'test')
        scenes=['weather_light','weather_medium'] if group=='light' else ['weather_heavy']
        for scene in scenes:
            out=a.out/scene;out.mkdir()
            row=next(r for r in data.records if r['scene_id']==scene)
            names=['GT','原半网格九帧','半网格末帧替代','四分之一九帧','四分之一末帧替代']
            values={name:[] for name in names}
            video=out/'full120_history_intervention.mp4'
            writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','6400x1080','-r','12','-i','-',
                '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
            font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
            with torch.inference_mode():
                try:
                    for frame in range(120):
                        rr=dict(row,frame_id=frame)
                        x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
                        current=x[:,-1:].expand_as(x)
                        c,_=data.context_for(rr);c=c[None].cuda();box=box_for(rr).cuda()
                        gt=np.array(Image.open((Path(config['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32)
                        outputs=[gt]+[(m.native(inp,c,box).clamp(0,1)[0,0]*255).cpu().numpy() for m,inp in [(base,x),(base,current),(compact,x),(compact,current)]]
                        canvas=Image.new('RGB',(6400,1080),'#161b22');draw=ImageDraw.Draw(canvas)
                        for i,(name,image) in enumerate(zip(names,outputs)):
                            values[name].append(image[240:600].copy())
                            canvas.paste(Image.fromarray(np.rint(image.clip(0,255)).astype(np.uint8)).convert('RGB'),(1280*i,56))
                            draw.text((1280*i+12,10),f'{scene} | {name} | {frame:03d}',font=font,fill='white')
                        writer.stdin.write(np.asarray(canvas).tobytes())
                        if frame%20==0:print('HISTORY',scene,frame,flush=True)
                finally:writer.stdin.close();assert writer.wait()==0
            values={k:np.stack(v) for k,v in values.items()}
            metrics=evaluate_signals(values,out)
            weights=base.quarter.front.first.weight.detach().float().abs().sum((0,2,3)).cpu().numpy()
            weights/=weights.sum()
            record={'base_checkpoint':str(base_path),'base_sha256':sha(base_path),'compact_sha256':sha(PREVIOUS/(group+'_temporal')/'best.pt'),
                    'history_lags':list(range(-8,1)),'base_front_absolute_weight_share':weights.tolist(),
                    'full_video':str(video),'video_sha256':sha(video),'summary':metrics['summary'],
                    'valid_scoring_ROI':row['eval_crop_tlhw'],'intervention':'replace historical normalized frames by the same current frame; reference and causal RAW corrections remain unchanged'}
            manifest['scenes'][scene]=record
            (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('RESULT',scene,json.dumps(record),flush=True)
    print('HISTORY_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
