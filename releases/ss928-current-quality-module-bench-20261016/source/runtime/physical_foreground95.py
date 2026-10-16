"""Keep independent92 background and restore compact physical RAW foreground."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from prepare_residual50 import CACHE50
from prepare_gate80 import CACHE80
from residual50 import RawSignal50
from physical_reference90 import FrozenTone90,detail,clean_common
from compact_match93 import components93,TOP,BOTTOM,CACHE92
from road_residual53 import support53
from fit_residual50 import fit
from repair_motion15 import load_captures

PHYS95=Path('/tmp/rawir-physical-foreground95-20261014')
CACHE95=Path('/tmp/rawir-residual95-20261014')

def prepare():
    scene='weather_heavy_c32';raw=RawSignal50(scene)
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    center=road>.9;center[:,:64]=False
    for path in sorted((DEST/scene/'train').glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text());row=meta['row']
        raw.for_record(row,0,(TOP,0,BOTTOM-TOP,1280));physical=raw.open[row['raw']['path']]
        out=PHYS95/path.name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        result=np.lib.format.open_memmap(out/'fore95.npy',mode='w+',dtype=np.float32,
                                       shape=(len(meta['frames']),BOTTOM-TOP,1280));result[:]=0
        tone=None
        for i,f in enumerate(meta['frames']):
            if f<16:continue
            if tone is None:
                initial=np.median(np.asarray(physical[f-16:f],np.float32),axis=0)
                tone=FrozenTone90(initial);reference=initial[TOP:BOTTOM]
            background=np.median(np.asarray(physical[max(0,f-64):f,TOP:BOTTOM],np.float32),axis=0)
            background=clean_common(background,reference)
            current=clean_common(physical[f,TOP:BOTTOM].astype(np.float32),background)
            fore=tone.apply(detail(current))-tone.apply(detail(background))
            fore-=np.median(fore[center]);result[i]=fore
            if i%20==0:print('PHYSICAL_FORE95',path.name,f,flush=True)
        result.flush()
        (out/'complete.json').write_text(json.dumps(dict(frames=meta['frames'],raw_path=row['raw']['path'],
            GT_used=False,initialization='past16',RAW_background='past64, excludes current',tone='frozen90',foreground_mask_applied=False)))

def calibrate():
    numerator=0.;denominator=0.;records=0;pixels=0
    road=support53((TOP,256,BOTTOM-TOP,768),'cpu')[0,0].numpy()>.9
    for path in sorted((DEST/'weather_heavy_c32'/'train').glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text())
        labels=torch.load(ROOT/'BODY_LABEL83-20261014'/path.name/'labels.pt',map_location='cpu',weights_only=False)
        gt=np.load(path/'GT.npy',mmap_mode='r')
        # Every GT-derived statistic is restricted to the original training ROI.
        g=gt[:,TOP:BOTTOM,256:1024].astype(np.float32)
        physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r')
        for i,f in enumerate(meta['frames']):
            if f<16 or i%7:continue
            background=np.median(g[max(0,i-64):i],axis=0)
            actual=g[i]-background;actual-=np.median(actual[road])
            for region in labels['regions'].get(f,[]):
                t=np.asarray(region['target'],int)
                if not 2<=len(t)<=16:continue
                y,x=t[:,0]-TOP,t[:,1]-256
                if np.any(y<0) or np.any(y>=BOTTOM-TOP) or np.any(x<0) or np.any(x>=768):
                    raise ValueError('GT calibration outside original training ROI')
                observed=physical[i,t[:,0]-TOP,t[:,1]].astype(np.float64)
                truth=actual[y,x].astype(np.float64)
                numerator+=float(np.sum(observed*truth));denominator+=float(np.sum(observed*observed))
                records+=1;pixels+=len(t)
    if denominator<=0:raise ValueError('No train compact foreground observations')
    gain=float(np.clip(numerator/denominator,.25,4))
    report=dict(gain=gain,records=records,pixels=pixels,fit_split='train',GT_ROI=[0,256,1024,768],
                GT_outside_train_ROI_used=False,validation_used=False,test_used=False,
                foreground='current GT - available past64 GT, original ROI only',labels='BODY_LABEL83 <=16 pixels')
    out=ROOT/'MOTION95-20261014'/'weather_heavy_c32';out.mkdir(parents=True,exist_ok=True)
    (out/'response.json').write_text(json.dumps(report,indent=2));print('GAIN95',json.dumps(report),flush=True)
    return gain

def infer(gain):
    road=support53((TOP,0,BOTTOM-TOP,1280),'cpu')[0,0].numpy()
    center=road>.9;center[:,:64]=False
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text())
            physical_meta=json.loads((PHYS95/path.name/'complete.json').read_text())
            if meta['frames']!=physical_meta['frames'] or meta['row']['raw']['path']!=physical_meta['raw_path']:
                raise ValueError('Physical input sequence differs')
            source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            parent=np.load(CACHE92/split/path.name/'total92.npy',mmap_mode='r')
            images=source[:,TOP:BOTTOM].astype(np.float32)+parent[:,TOP:BOTTOM]
            physical=np.load(PHYS95/path.name/'fore95.npy',mmap_mode='r')
            gate=np.load(CACHE80/split/path.name/'signed_gate80.npy',mmap_mode='r')
            out=CACHE95/split/path.name;out.mkdir(parents=True,exist_ok=True)
            result=np.lib.format.open_memmap(out/'delta95.npy',mode='w+',dtype=np.float32,shape=source.shape);result[:]=0
            for i,f in enumerate(meta['frames']):
                if f<16:continue
                background=np.median(images[max(0,i-64):i],axis=0)
                foreground=images[i]-background;foreground-=np.median(foreground[center])
                signed=gate[i,TOP:BOTTOM].astype(np.float32)
                _,confirmed=components93(signed,road)
                mask=(confirmed!=0)&(physical[i]*signed>0)
                result[i,TOP:BOTTOM]=np.clip(gain*physical[i]-foreground,-8,8)*np.abs(signed)*mask*road
                if i%20==0:print('FORE95_INFER',split,path.name,f,flush=True)
            result.flush();(out/'complete.json').write_text(json.dumps(dict(GT_used_for_inference=False,parent=92,
                gain_fit='original train ROI',background='independent92',compact_confirmation='physicalRAW93')))

def select():
    fit(95,road=True,parent_delta=92,parent_file='total92.npy')
    path=ROOT/'MOTION95-20261014'/'weather_heavy_c32'/'fit.json'
    report=json.loads(path.read_text());settings=[r['alpha'] for r in report['candidates']]
    body={a:[] for a in settings};tiny={a:[] for a in settings}
    for c in load_captures('weather_heavy_c32',raw_targets=True):
        cap='capture_%02d'%c['meta']['capture']
        labels=torch.load(ROOT/'BODY_LABEL83-20261014'/cap/'labels.pt',map_location='cpu',weights_only=False)
        source=np.load(CACHE50/'train'/cap/'source43.npy',mmap_mode='r')
        parent=np.load(CACHE92/'train'/cap/'total92.npy',mmap_mode='r')
        delta=np.load(CACHE95/'train'/cap/'delta95.npy',mmap_mode='r')
        for i,f in enumerate(c['meta']['frames']):
            if i not in c['valid']:continue
            image=np.clip(source[i].astype(np.float32)+parent[i],0,255);gt=c['arrays']['GT'][i]
            for region in labels['regions'].get(f,[]):
                t=np.asarray(region['target'],int);ring=np.asarray(region['ring'],int)
                truth=float(gt[t[:,0],t[:,1]].astype(np.float32).mean()-np.median(gt[ring[:,0],ring[:,1]].astype(np.float32)))
                for alpha in settings:
                    target=np.clip(image[t[:,0],t[:,1]]+alpha*delta[i,t[:,0],t[:,1]],0,255)
                    back=np.clip(image[ring[:,0],ring[:,1]]+alpha*delta[i,ring[:,0],ring[:,1]],0,255)
                    error=abs(float(target.mean()-np.median(back))-truth)
                    body[alpha].append(error)
                    if len(t)<=16:tiny[alpha].append(error)
    whole={a:float(np.mean(v)) for a,v in body.items()};small={a:float(np.mean(v)) for a,v in tiny.items()}
    report['strict_selection']=report['best'];zero=report['candidates'][0]
    for row in report['candidates']:
        alpha=row['alpha'];row['body_gap_error']=whole[alpha];row['tiny_body_gap_error']=small[alpha]
        row['strict_eligible']=row['eligible']
        row['eligible']=(all(row['metrics'][k]<=report['source'][k]+.05 for k in ['gap','ghost','signed_ghost']) and
                         all(q<=1.005*p for q,p in zip(row['capture_flicker'],zero['capture_flicker'])) and
                         whole[alpha]<=whole[0.]+1e-6 and small[alpha]<=small[0.]+1e-6)
        row['score']+=4*whole[alpha]/max(whole[0.],.05)+6*small[alpha]/max(small[0.],.05)
    report['best']=min([r for r in report['candidates'] if r['eligible']],key=lambda r:r['score'])
    report['diagnostic_best']=min(report['candidates'],key=lambda r:r['score'])
    report.update(status='passed' if report['best']['alpha']>0 else 'no_pass',body_source_gap_error=whole[0.],
                  tiny_body_source_gap_error=small[0.],body_labels_reference='BODY_LABEL83 original train ROI',
                  tolerance_non_flicker_gray=.05,validation_used=False,test_used=False)
    path.write_text(json.dumps(report,indent=2));print('FORE95_FIT',json.dumps(report),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(2);torch.set_num_threads(2);prepare();gain=calibrate();infer(gain);select()
