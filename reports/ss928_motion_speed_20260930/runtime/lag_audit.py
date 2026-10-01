"""Check RAW/GT and model/GT time correspondence on GT-derived moving-road pixels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup
from history_probe import foreground


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    report={'meaning':'positive best offset: comparing a later model frame to current GT gives higher correlation; diagnostic aggregate, not a per-target delay estimate',
            'GT_not_shifted_for_training':True,'scenes':{}}
    for group,scenes in [('light',['weather_light','weather_medium']),('heavy',['weather_heavy'])]:
        _,base,config,teacher,base_path,df,box_for=setup(group);data=df(config,'test')
        for scene in scenes:
            directory=a.root/('history_probe' if group=='light' else 'history_heavy')/scene
            cache=np.load(directory/'road_signals_and_masks.npz',allow_pickle=False)
            masks_path=a.root/('signed_targets' if group=='light' else 'signed_targets_heavy')/scene/'GT_signed_target_masks.npz'
            masks=np.load(masks_path,allow_pickle=False)['current']
            pixels=np.any(masks[16:104],axis=0)
            gt=cache['GT'][16:104,pixels].astype(np.float64).reshape(-1);gt-=gt.mean()
            raw=[];row=next(r for r in data.records if r['scene_id']==scene)
            for frame in range(120):
                rr=dict(row,frame_id=frame)
                x=data.normalized_stack(rr,(0,0,1024,1280))[-1].numpy()*255
                raw.append(x[240:600].copy())
            raw=np.stack(raw)
            road=np.zeros(pixels.shape,bool)
            y,x=np.mgrid[:360,:1280]
            road=(y>=15+230*x/1279)&(y<=100+230*x/1279)
            signal,background,drift=foreground(raw,road)
            methods={'当前RAW':signal,**{k:cache[k] for k in cache.files if k not in ['GT','GT_current_mask']}}
            record={'motion_pixel_union':int(pixels.sum()),'frames':[16,103],'methods':{}}
            for name,images in methods.items():
                scores={}
                for offset in range(-8,9):
                    pred=images[16+offset:104+offset,pixels].astype(np.float64).reshape(-1);pred-=pred.mean()
                    scores[str(offset)]=float(np.dot(gt,pred)/max(np.linalg.norm(gt)*np.linalg.norm(pred),1e-12))
                best=max(scores,key=scores.get)
                record['methods'][name]={'best_offset_frames':int(best),'zero_offset_correlation':scores['0'],'best_correlation':scores[best],'all_offsets':scores}
            report['scenes'][scene]=record
            np.savez_compressed(a.out/(scene+'_current_RAW_signal.npz'),signal=signal)
            (a.out/'lag_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print('LAG_AUDIT',scene,json.dumps(record['methods'],ensure_ascii=False),flush=True)
    print('LAG_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
