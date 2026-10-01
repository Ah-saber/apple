"""Refine cached fixed-weight controls for bright and dark GT targets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, binary_dilation, label, find_objects


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    report={'NPU_measured':False,'analysis_modifies_weights':False,'mask_source':'GT only, signed targets; native road; abs contrast>10 after sigma .7; area4..600; box<=60x35',
            'signal_source':'floating point model output, each method 120-frame median background removed; per-frame scalar road drift removed',
            'video_used_for_metrics':False,'scenes':{}}
    for directory in sorted(a.root.glob('weather_*')):
        cache=directory/'road_signals_and_masks.npz'
        if not cache.exists():continue
        data=np.load(cache,allow_pickle=False)
        values={k:data[k] for k in data.files if k!='GT_current_mask'}
        truth=values['GT'];h,w=truth.shape[-2:]
        road=np.zeros((h,w),np.uint8)
        cv2.fillPoly(road,[np.array([[0,15],[1279,245],[1279,330],[0,100]])],1);road=road.astype(bool)
        masks=[];boxes=[]
        for image in truth:
            smooth=gaussian_filter(image,.7);kept=np.zeros((h,w),bool);records=[]
            for sign in [1,-1]:
                components,_=label((sign*smooth>10)&road)
                for i,slices in enumerate(find_objects(components),1):
                    if slices is None:continue
                    yy,xx=slices;area=int((components[slices]==i).sum())
                    if 4<=area<=600 and xx.stop-xx.start<=60 and yy.stop-yy.start<=35:
                        kept[slices]|=components[slices]==i
                        records.append([yy.start,xx.start,yy.stop-yy.start,xx.stop-xx.start,area,sign])
            masks.append(kept);boxes.append(records)
        masks=np.stack(masks);rows=[]
        for t in range(8,len(truth)):
            current=masks[t]
            history=np.any(masks[t-8:t],axis=0)
            old=binary_dilation(history,iterations=1)&~binary_dilation(current,iterations=3)&road
            past_signal=np.where(masks[t-8:t],truth[t-8:t],0)
            strongest=np.argmax(np.abs(past_signal),axis=0)
            polarity=np.sign(np.take_along_axis(past_signal,strongest[None],axis=0)[0])
            # Extend the historical sign to its one-pixel neighborhood.
            positive=binary_dilation(polarity>0,iterations=1)
            negative=binary_dilation(polarity<0,iterations=1)
            polarity=np.where(positive&~negative,1,np.where(negative&~positive,-1,0))
            old&=polarity!=0
            gt=truth[t];change=gt-truth[t-1]
            static=road&(np.abs(change)<=1)&~binary_dilation(current|history,iterations=3)
            row={'frame':t,'current_pixels':int(current.sum()),'historical_only_pixels':int(old.sum()),'GT_boxes':boxes[t],'methods':{}}
            for name,pred in values.items():
                if name=='GT':continue
                image=pred[t];r={}
                if current.any():
                    r['current_target_signed_contrast_ratio']=float((np.sign(gt[current])*image[current]).sum()/np.abs(gt[current]).sum())
                    r['current_target_signal_mae_gray']=float(np.abs(image[current]-gt[current]).mean())
                if old.any():
                    r['historical_only_signed_excess_gray']=float(np.maximum(polarity[old]*(image[old]-gt[old]),0).mean())
                    r['historical_only_signal_mae_gray']=float(np.abs(image[old]-gt[old]).mean())
                if static.any():r['road_static_residual_change_gray']=float(np.abs((image-pred[t-1])-change)[static].mean())
                errors=[]
                for yy,xx,hh,ww,area,sign in boxes[t]:
                    top=max(0,yy-10);left=max(0,xx-16);bottom=min(h,yy+hh+10);right=min(w,xx+ww+16)
                    g=np.maximum(sign*gt[top:bottom,left:right]-5,0)
                    q=np.maximum(sign*image[top:bottom,left:right]-5,0)
                    if g.sum()>0 and q.sum()>0:
                        y,x=np.mgrid[:g.shape[0],:g.shape[1]]
                        center_gt=np.array([(y*g).sum(),(x*g).sum()])/g.sum()
                        center_pred=np.array([(y*q).sum(),(x*q).sum()])/q.sum()
                        errors.append(float(np.linalg.norm(center_pred-center_gt)))
                if errors:r['local_target_centroid_error_pixels']=float(np.mean(errors))
                row['methods'][name]=r
            rows.append(row)
        summary={}
        for name in values:
            if name=='GT':continue
            fields=set(k for row in rows for k in row['methods'][name])
            summary[name]={k:float(np.mean([row['methods'][name][k] for row in rows if k in row['methods'][name]])) for k in fields}
        dest=a.out/directory.name;dest.mkdir()
        record={'summary':summary,'per_frame':rows,'bright_counts':[sum(b[-1]==1 for b in row) for row in boxes],
                'dark_counts':[sum(b[-1]==-1 for b in row) for row in boxes],
                'valid_ROI':'native road within eval crop for light/medium/heavy; C32 excluded',
                'limits':'GT threshold-assisted evaluation, not human target annotations; road static statistic excludes target histories; whole-image quality not measured by this probe'}
        (dest/'signed_target_results.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
        np.savez_compressed(dest/'GT_signed_target_masks.npz',current=masks)
        # Display exact locations from GT; pictures decoded from review video, metrics stay floating point.
        ranking=sorted(range(8,len(truth)-4),key=lambda t:float(np.abs(truth[t][masks[t]]).sum()),reverse=True)
        chosen=[]
        for t in ranking:
            if boxes[t] and all(abs(t-v)>12 for v in chosen):chosen.append(t)
            if len(chosen)==3:break
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        for central in chosen:
            yy,xx,hh,ww,area,sign=max(boxes[central],key=lambda b:b[4])
            top=max(0,min(h-80,yy+hh//2-40));left=max(0,min(w-128,xx+ww//2-64))
            video=cv2.VideoCapture(str(directory/'full120_history_intervention.mp4'));video.set(cv2.CAP_PROP_POS_FRAMES,central-3)
            canvas=Image.new('RGB',(len(values)*384,2144),'#161b22');draw=ImageDraw.Draw(canvas)
            for row,t in enumerate(range(central-3,central+5)):
                ok,image=video.read();assert ok,(directory,t)
                for col,name in enumerate(values):
                    patch=image[56+240+top:56+240+top+80,1280*col+left:1280*col+left+128]
                    patch=Image.fromarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)).resize((384,240),Image.Resampling.NEAREST)
                    canvas.paste(patch,(col*384,row*268+28));draw.text((col*384+4,row*268+3),f'{name} | {t:03d}',font=font,fill='white')
            video.release();canvas.save(dest/f'target_{central:03d}_eight_frames.png')
        report['scenes'][directory.name]=record
        (a.out/'signed_target_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print('SIGNED_TARGET_RESULT',directory.name,json.dumps(summary,ensure_ascii=False),flush=True)
    print('SIGNED_TARGET_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
