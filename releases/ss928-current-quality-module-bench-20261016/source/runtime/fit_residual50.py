"""Select residual strength on the unchanged C32 training regions."""
import json
import numpy as np
import cv2
import torch
from repair_motion15 import load_captures,tensor_at
from prepare_motion14 import DEST
from prepare_residual50 import OUT50,CACHE50
from residual50 import Residual50,RawSignal50

def infer(number=50,model_class=Residual50,parent_delta=None):
    from pathlib import Path
    outroot=OUT50.parent/('MOTION%d-20261014'%number)
    cacheroot=Path('/tmp/rawir-residual%d-20261014'%number)
    ck=torch.load(outroot/'weather_heavy_c32'/'last_step500.pt',map_location='cpu',weights_only=False)
    model=model_class().cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);raw=RawSignal50('weather_heavy_c32')
    for split in ['train','val_continuous']:
        for path in sorted((DEST/'weather_heavy_c32'/split).glob('capture_*')):
            cache=cacheroot/split/path.name;cache.mkdir(parents=True,exist_ok=True)
            if (cache/'delta_complete.json').exists():continue
            meta=json.loads((path/'complete.json').read_text());source=np.load(CACHE50/split/path.name/'source43.npy',mmap_mode='r')
            parent=np.load(Path('/tmp/rawir-residual%d-20261014'%parent_delta)/split/path.name/('delta%d.npy'%parent_delta),mmap_mode='r') if parent_delta else None
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['baseline','current_only','gated_history','past_background','activity','occupancy']}
            delta=np.lib.format.open_memmap(cache/('delta%d.npy'%number),mode='w+',dtype=np.float32,shape=source.shape)
            previous=None
            for i,f in enumerate(meta['frames']):
                v={k:torch.from_numpy(a[i].astype(np.float32))[None,None].cuda() for k,a in arrays.items()}
                v['raw50']=raw.for_record(meta['row'],f,(0,0,1024,1280))
                image=source[i] if parent is None else (source[i]+parent[i]).clip(0,255)
                s=torch.from_numpy(image.copy())[None,None].cuda()
                with torch.inference_mode():d=model(v,s,s if previous is None else previous)
                delta[i]=d[0,0].cpu().numpy();previous=s
            delta.flush();(cache/'delta_complete.json').write_text(json.dumps(dict(frames=len(meta['frames']),fit_split='train',source='QUALITY43')))
            print('RESIDUAL%d_INFER'%number,split,path.name,flush=True)

def fit(number=50,road=False,parent_delta=None,parent_file=None):
    from pathlib import Path
    outroot=OUT50 if number==50 else OUT50.parent/('MOTION%d-20261014'%number)
    delta_root=CACHE50 if number==50 else Path('/tmp/rawir-residual%d-20261014'%number)
    captures=load_captures('weather_heavy_c32',raw_targets=True);settings=[0.,.1,.25,.5,.75,1.]
    if road:
        from road_residual53 import focus53
        captures=[focus53(c) for c in captures]
    keys=['gap','inversions','ghost','signed_ghost','flicker']
    stats={a:{k:[] for k in keys} for a in settings};capture_flicker={a:[] for a in settings}
    for c in captures:
        crop=c['saved']['roi'];y,x,h,w=crop;cache=CACHE50/'train'/('capture_%02d'%c['meta']['capture'])
        sources=np.load(cache/'source43.npy',mmap_mode='r');deltas=np.load(delta_root/'train'/cache.name/('delta%d.npy'%number),mmap_mode='r');previous=None;last_gt=None;local_flicker={a:[] for a in settings}
        parent=np.load(Path('/tmp/rawir-residual%d-20261014'%parent_delta)/'train'/cache.name/(parent_file or ('delta%d.npy'%parent_delta)),mmap_mode='r') if parent_delta else None
        for i,f in enumerate(c['meta']['frames']):
            source=sources[i,y:y+h,x:x+w];delta=deltas[i,y:y+h,x:x+w]
            if parent is not None:source=(source+parent[i,y:y+h,x:x+w]).clip(0,255)
            gt=c['arrays']['GT'][i,y:y+h,x:x+w].astype(np.float32)
            images={a:(source+a*delta).clip(0,255) for a in settings}
            if i>=8:
                for r in c['saved']['regions'].get(f,[]):
                    t=np.asarray(r['target'],int)-[y,x];ring=np.asarray(r['ring'],int)-[y,x]
                    if not 1<=len(t)<=9 or len(ring)<3:continue
                    if not (np.all(t>=8) and np.all(ring>=8) and np.all(t[:,0]<h-8) and np.all(t[:,1]<w-8) and np.all(ring[:,0]<h-8) and np.all(ring[:,1]<w-8)):continue
                    true=float(gt[t[:,0],t[:,1]].mean()-np.median(gt[ring[:,0],ring[:,1]]))
                    for a,image in images.items():
                        gap=float(image[t[:,0],t[:,1]].mean()-np.median(image[ring[:,0],ring[:,1]]))
                        stats[a]['gap'].append(abs(gap-true));stats[a]['inversions'].append(gap*true<0)
                ghost=np.asarray(c['saved']['labels'].get(f,{}).get('ghost',[]),int).reshape(-1,3)
                if len(ghost):
                    gy,gx=ghost[:,0]-y,ghost[:,1]-x;valid=(gy>=8)&(gy<h-8)&(gx>=8)&(gx<w-8)
                    if valid.any():
                        for a,image in images.items():
                            e=image-gt;e-=cv2.blur(e,(21,21),borderType=cv2.BORDER_REFLECT)
                            stats[a]['ghost'].extend(np.abs(e[gy[valid],gx[valid]]).tolist())
                            stats[a]['signed_ghost'].extend(np.maximum(e[gy[valid],gx[valid]]*ghost[valid,2],0).tolist())
                if last_gt is not None:
                    mask=cv2.dilate((np.abs(gt-last_gt)>3).astype(np.uint8),np.ones((7,7),np.uint8))==0
                    mask[:8]=False;mask[-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
                    for a,image in images.items():
                        change=(image-previous[a])[mask]
                        if change.size:
                            v=float(np.abs(change-np.median(change)).mean());stats[a]['flicker'].append(v);local_flicker[a].append(v)
            previous=images;last_gt=gt
        for a in settings:capture_flicker[a].append(float(np.mean(local_flicker[a])))
    means={a:{k:float(np.mean(v)) if v else None for k,v in values.items()} for a,values in stats.items()}
    source=means[0.];rows=[]
    for a,m in means.items():
        eligible=all(m[k]<=source[k]+1e-6 for k in ['gap','ghost','signed_ghost'] if m[k] is not None) and all(q<=1.005*p for q,p in zip(capture_flicker[a],capture_flicker[0.]))
        score=sum(weight*m[k]/max(source[k],.05) for k,weight in [('gap',4),('ghost',2),('signed_ghost',2),('flicker',1)] if m[k] is not None)
        rows.append(dict(alpha=a,metrics=m,capture_flicker=capture_flicker[a],eligible=eligible,score=score))
    best=min([r for r in rows if r['eligible']],key=lambda r:r['score']);diagnostic=min(rows,key=lambda r:r['score'])
    report=dict(scene='weather_heavy_c32',parent_delta=parent_delta,parent_file=parent_file,source=source,candidates=rows,best=best,diagnostic_best=diagnostic,status='passed' if best['alpha']>0 else 'no_pass',fit_split='train',validation_used=False,test_used=False,training_ROIs=[c['saved']['roi'] for c in captures],flicker_limit='each train capture <= 1.005 source')
    out=outroot/'weather_heavy_c32';out.mkdir(parents=True,exist_ok=True)
    (out/'fit.json').write_text(json.dumps(report,indent=2));print('RESIDUAL%d_FIT'%number,json.dumps(report),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory);infer();fit()
