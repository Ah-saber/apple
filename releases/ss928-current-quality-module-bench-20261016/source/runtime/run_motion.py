"""Per-history peak candidate. Preserve scene rules; skip SHA checks."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/data/zhangbenzhuang/huawei_sr')
sys.path.insert(0, str(ROOT/'runs/SS928-BOARD-FEEDBACK-ROUND3-20261001/runtime'))
from run_round3 import (GROUPS, R2, V13, load_group, night_source, ShortHistory,
                        BlendedModel, BlendedNightFront, NightFixed, BoardHalfOptimized, LastFrames)
from candidate_ops import HistoryGate
from run_speed import measure, export_selected, frozen as structural


def evaluate_motion(group,out,night=False,detector='peak',export=False,timing=True,warm=50,steps=300):
    import subprocess
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    from run_compact import temporal_metrics
    report={'NPU_measured':False,'SDK_compiled':False,'packaged':False,'pushed':False,
            'selection':'training RAW percentile; validation PSNR guard; source fallback if all proposals fail; no test used for threshold selection',
            'mechanism':'per-history pointwise difference; peak or mean spatial detector; unchanged current; single preserved temporal convolution',
            'scenes':{}}
    HistoryGate.default_count=9 if night else 3
    HistoryGate.default_detector=detector
    HistoryBlend=HistoryGate
    scene_names=['night_ordinary','night_special'] if night else GROUPS[group]['scenes']
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
    for scene in scene_names:
        dest=out/scene;dest.mkdir()
        if night:
            kind=scene.removeprefix('night_');source=night_source(kind)
            from ir_sr.training import dataset_for_config
            manifest=json.loads((V13/'night_calibration_final'/scene/'manifest.json').read_text());config=manifest['config'];df=dataset_for_config
            ckpath=None;caldir=V13/'night_calibration_final'/scene
        else:
            _,base,config,_,_,df,box_for=load_group(group)
            source=ShortHistory(base,3,'half');ckpath=R2/f'half3_{group}_3/best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);source.load_state_dict(ck['model'])
            source=source.cuda().float().eval();caldir=V13/'deployment_corrected'/scene
        training=[]
        with torch.inference_mode():
            probe=HistoryBlend(1).cuda()
            for path in sorted(caldir.glob('train_*.npz')):
                x=torch.from_numpy(np.load(path,allow_pickle=False)['nine_raw']).cuda().float()
                training.append(probe.signal(x)[:,:,3:-3,3:-3].cpu().numpy().ravel())
        assert len(training)==4
        signals=np.concatenate(training);thresholds={p:float(np.percentile(signals,p)) for p in ([95,99] if detector=='peak' else [80,90])}
        proposals={}
        blends={}
        for p,strength in ([(95,.75),(99,1.)] if detector=='peak' else [(80,.25),(90,.5)]):
            blend=HistoryBlend(thresholds[p],strength,9 if night else 3).cuda();name=f'p{p}_s{strength}'
            blends[name]=blend
            if night:
                candidate=copy.deepcopy(source);candidate.front=BlendedNightFront(source.front,blend).cuda().eval()
            else:candidate=BlendedModel(source,blend).cuda().eval()
            proposals[name]=candidate
        def inputs(data,row):
            x=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            c,_=data.context_for(row,**({'crop_tlhw':(0,0,1024,1280)} if night else {}));c=c[None].cuda()
            if night:x=x.half() if kind=='ordinary' else x
            return x,c
        def prediction(model,x,c,box=None):
            return (model(x,c)[0,0,::3,::3].float() if night else model.native(x,c,box)[0,0].float().clamp(0,1)*255).cpu().numpy()
        val=df(config,'val');vrows=[r for r in val.records if r['scene_id']==scene][:12];assert vrows
        validation={n:[] for n in ['source']+list(proposals)}
        with torch.inference_mode():
            for row in vrows:
                x,c=inputs(val,row);box=None if night else box_for(row).cuda()
                gt=np.asarray(Image.open(Path(config['data_root'])/row['target']['path']),dtype=np.float32)
                yy,xx,hh,ww=row['eval_crop_tlhw'];yy+=3;xx+=3;hh-=6;ww-=6
                for name,model in {'source':source,**proposals}.items():
                    pred=prediction(model,x,c,box);e=pred[yy:yy+hh,xx:xx+ww]-gt[yy:yy+hh,xx:xx+ww]
                    validation[name].append(float(-10*np.log10(max(float((e**2).mean())/255**2,1e-12))))
        scores={n:float(np.mean(v)) for n,v in validation.items()};eligible=[n for n in proposals if scores[n]>=scores['source']-.05]
        chosen=eligible[0] if eligible else 'source';candidate=proposals.get(chosen,source)
        data=df(config,'test');row=next(r for r in data.records if r['scene_id']==scene)
        from ir_sr.sequence_normalization import regional_key
        norm=data.base.sequence_normalization;key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id'];count=int(norm.by_key[key]['frames'])
        models={'原夜间九帧' if night else '上轮半网格三帧':source,'输入局部当前':candidate}
        names=['GT']+list(models);width=len(names)*1280;video=dest/f'full{count}_comparison.mp4'
        writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x1080','-r','12','-i','-',
            '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        if night:
            roi_y,height=(0,720) if kind=='ordinary' else (680,344)
            polygon=([[0,80],[80,208],[175,336],[270,387],[600,452],[900,526],[1279,657],[1279,710],[880,570],[550,480],[260,416],[155,363],[70,250],[0,120]] if kind=='ordinary' else [[0,70],[1279,282],[1279,344],[0,132]])
        else:
            roi_y,height=(0,650) if scene=='day_normal' else (240,360)
            polygon=([[350,379],[450,406],[600,438],[800,486],[1000,539],[1279,607],[1279,621],[1000,552],[800,505],[600,456],[450,421],[350,394]] if scene=='day_normal' else [[0,15],[1279,245],[1279,330],[0,100]])
        road=np.zeros((height,1280),np.uint8);cv2.fillPoly(road,[np.array(polygon)],1);road=road.astype(bool)
        yy,xx,hh,ww=row['eval_crop_tlhw'];coord_y,coord_x=np.mgrid[:height,:1280]
        road&=(coord_y+roi_y>=yy+3)&(coord_y+roi_y<yy+hh-3)&(coord_x>=xx+3)&(coord_x<xx+ww-3)
        roads={n:[] for n in names};records=[];previous=None;histories=[]
        with torch.inference_mode():
            try:
                for frame in range(count):
                    rr=dict(row,frame_id=frame);x,c=inputs(data,rr);box=None if night else box_for(rr).cuda()
                    ids=[int(v) for v in data.frame_ids(rr)];assert max(ids)<=frame;histories.append(ids)
                    gt=np.asarray(Image.open((Path(config['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32)
                    images=[gt]+[prediction(m,x,c,box) for m in models.values()]
                    canvas=Image.new('RGB',(width,1080),'#161b22');draw=ImageDraw.Draw(canvas);item={'frame':frame,'methods':{}}
                    y,x0,h,w=rr['eval_crop_tlhw'];y+=3;x0+=3;h-=6;w-=6;g=gt[y:y+h,x0:x0+w]
                    for col,(name,image) in enumerate(zip(names,images)):
                        canvas.paste(Image.fromarray(np.rint(image.clip(0,255)).astype(np.uint8)).convert('RGB'),(1280*col,56));draw.text((1280*col+12,10),f'{scene} | {name} | {frame:03d}',font=font,fill='white')
                        if scene!='weather_heavy_c32':roads[name].append(image[roi_y:roi_y+height].copy())
                        if col:
                            pred=image[y:y+h,x0:x0+w];e=pred-g;r={'psnr_db':float(-10*np.log10(max(float((e**2).mean())/255**2,1e-12))), 'mae_gray':float(np.abs(e).mean()),'bias_gray':float(e.mean())}
                            if previous:r.update(temporal_metrics(pred,previous[col][y:y+h,x0:x0+w],g,previous[0][y:y+h,x0:x0+w]))
                            item['methods'][name]=r
                    writer.stdin.write(np.asarray(canvas).tobytes());records.append(item);previous=images
                    if frame in [30,45]:canvas.save(dest/f'frame_{frame:03d}.png')
                    if frame%30==0:print('INPUT_BLEND_EVAL',scene,frame,flush=True)
            finally:writer.stdin.close();assert writer.wait()==0
        if scene!='weather_heavy_c32':
            cache={}
            for name,v in roads.items():
                v=np.stack(v);v=v-np.median(v,axis=0);v=v-np.median(v[:,road],axis=1)[:,None,None];cache[name]=v
            np.savez_compressed(dest/'road_signals_and_masks.npz',**cache)
            (dest/'target_region.json').write_text(json.dumps({'roi_y':roi_y,'roi_x':0,'polygon':polygon,'valid_native_tlhw':row['eval_crop_tlhw'],'metric_border_pixels':3,'selection':'same scene GT lane; actual valid crop; C32 excluded'}))
        exports={}; full_timing={}
        if timing or export:
            prepared=[]
            for path in sorted(caldir.glob('test_*.npz')):
                with np.load(path,allow_pickle=False) as sample:
                    values=tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw','reference_thumb'])
                if not night:values=tuple(v.half() for v in values)
                prepared.append(values)
            assert len(prepared)==12
            for label,model in [('source',source),('candidate',candidate)]:
                if night:
                    board=NightFixed(model,kind).cuda().eval()
                    structural.replace_samples(board,8)
                    board.output=structural.ChannelCopyOutput().cuda().half()
                else:
                    core=BoardHalfOptimized(source.source,box).cuda().half().eval()
                    structural.replace_samples(core,8)
                    core.rows=structural.ChannelCopyOutput().cuda().half()
                    board=LastFrames(core,3)
                    if label=='candidate' and chosen!='source':
                        board=BlendedModel(board,blends[chosen]).cuda().eval()
                if timing:
                    full_timing[label]=measure(board,prepared,warm,steps)
                if export and (label=='source' or chosen!='source'):
                    exports[label]=export_selected(board,prepared[-1],dest/(label+'_block8_copy.onnx'))
                del board
            if timing:
                full_timing['relative_to_source']={
                    method:{key:100*(1-full_timing['candidate'][method][key]/full_timing['source'][method][key])
                            for key in ['mean_ms','p95_ms']}
                    for method in ['gpu_event','full_synchronized_wall']}
        summary={name:{k:float(np.mean([v['methods'][name][k] for v in records if k in v['methods'][name]])) for k in records[-1]['methods'][name]} for name in models}

        report['scenes'][scene]={'frames':count,'validation_psnr':scores,'validation_rows':vrows,'thresholds':thresholds,'chosen':chosen,'validation_guard_passed':bool(eligible),'history_ids':histories,'all_causal':True,
            'summary':summary,'per_frame':records,'graphs':exports,'full_GPU_timing':full_timing,'detector':detector,'hash_checks':False,'automatically_adopted':False,'checkpoint':str(ckpath) if ckpath else 'frozen night_source','video':str(video),
            'calibration_sources':[{'path':str(p)} for p in sorted(caldir.glob('train_*.npz'))]}
        (out/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print('INPUT_BLEND_COMPLETE',scene,json.dumps(summary,ensure_ascii=False),flush=True)




def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--group",choices=["day","light","heavy"])
    group.add_argument("--night",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--detector",choices=["mean","peak"],default="peak")
    parser.add_argument("--export",action="store_true")
    parser.add_argument("--skip-timing",action="store_true")
    parser.add_argument("--warm",type=int,default=50)
    parser.add_argument("--steps",type=int,default=300)
    args=parser.parse_args()
    if args.steps<1 or args.warm<0:
        raise ValueError("Invalid timing counts")
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    evaluate_motion(args.group,args.out,args.night,args.detector,args.export,
                    not args.skip_timing,args.warm,args.steps)


if __name__=="__main__":
    main()
