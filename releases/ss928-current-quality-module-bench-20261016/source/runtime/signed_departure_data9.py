"""Restore training departure polarity and exclude current occupied target pixels."""
import json
import torch
from torch.nn import functional as F
from target_residual import ROOT
from prepare_native_targets import prepare
from target_training_guidance import balanced_raster

def build(scene):
    out=ROOT/('signed9_departure_'+scene);out.mkdir(exist_ok=False)
    cache=torch.load(ROOT/('velocity9_cache_'+scene)/'tiles.pt',map_location='cpu',weights_only=False)
    values=cache['values'];labels=prepare(scene)
    original={tuple(v['geometry']):v for v in values if 'training_history_stride' not in v}
    maps={};patches=[]
    totals=dict(tiles=len(values),augmented_tiles=0,changed_maps=0,negative_before=0,negative_after=0,current_overlap_before=0,current_overlap_after=0,polarity_changed_pixels=0)
    def target_map(frame):
        if frame not in maps:
            label=labels['labels'].get(frame)
            maps[frame]=torch.from_numpy(balanced_raster(label['target']).copy())[None,None] if label else torch.zeros(1,1,1024,1280)
        return maps[frame]
    for i,v in enumerate(values):
        augmented='training_history_stride' in v
        totals['augmented_tiles']+=int(augmented)
        frame,top,left,scale=v['geometry'];size=128*scale
        source=original[tuple(v['geometry'])] if augmented else v
        for prefix in ['', 'pair_']:
            before=v[prefix+'ghost']
            occupied=F.max_pool2d((v[prefix+'target'].abs()>0).float(),3,1,1)
            if augmented:
                past=torch.zeros_like(before)
                for f in v[prefix+'history_frames'][:-1]:
                    a=target_map(f)[...,top:top+size,left:left+size]
                    if a.shape[-2:]!=(size,size):raise ValueError('Training target geometry mismatch')
                    a=F.interpolate(a,size=before.shape[-2:],mode='area')
                    # History is chronological. The latest nonzero polarity owns overlap.
                    past=torch.where(a!=0,a,past)
                extra=past*(1-occupied)
                base=source[prefix+'ghost']
                after=torch.where(base.abs()>=extra.abs(),base,extra)*(1-occupied)
            else:after=before*(1-occupied)
            totals['negative_before']+=int((before<0).sum())
            totals['negative_after']+=int((after<0).sum())
            totals['current_overlap_before']+=int(((before!=0)&(occupied!=0)).sum())
            totals['current_overlap_after']+=int(((after!=0)&(occupied!=0)).sum())
            totals['polarity_changed_pixels']+=int(((before!=0)&(after!=0)&(before.sign()!=after.sign())).sum())
            if not torch.equal(before,after):
                patches.append(dict(tile=i,key=prefix+'ghost',value=after));totals['changed_maps']+=1
    report=dict(scene=scene,**totals,source_split='train',source_cache='velocity9_cache_'+scene,
        correction='Signed chronological historical GT targets; binary current occupancy; original signed departure retained; no current target support counted as departure',
        unchanged='RAW, GT values, current target/ring labels, reference field, inference inputs and evaluation rules',
        original_cache_retained=True,packaged=False,pushed=False)
    torch.save(dict(patches=patches,metadata=report),out/'patches.pt')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print('SIGNED9_DEPARTURE',json.dumps(report),flush=True)

def apply(scene,values):
    data=torch.load(ROOT/('signed9_departure_'+scene)/'patches.pt',map_location='cpu',weights_only=False)
    for patch in data['patches']:values[patch['tile']][patch['key']]=patch['value'].clone()
    return data['metadata']

if __name__=='__main__':
    from launch_compact_students import SCENES
    torch.set_num_threads(2)
    for scene in SCENES:build(scene)
