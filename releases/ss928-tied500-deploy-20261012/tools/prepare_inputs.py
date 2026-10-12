"""Convert the delivered exact FP16 NPZ inputs to compiler/board binary files."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SCENES=['night_ordinary','day_normal','weather_light','weather_medium','weather_heavy','weather_heavy_c32','night_special']

def run(scene,out):
    source=ROOT/'models'/scene;meta=json.loads((source/'manifest.json').read_text())
    folder=out/scene;folder.mkdir(parents=True,exist_ok=False)
    contracts=meta['graphs']['tied500']['inputs']
    if contracts!=meta['graphs']['retained_original']['inputs']:raise ValueError('Both roles must use identical inputs')
    records=[]
    for name in [r['file'] for r in meta['calibration']]+['smoke_inputs.npz']:
        dest=folder/Path(name).stem;dest.mkdir()
        with np.load(source/name) as values:
            inputs=[]
            for spec in contracts:
                a=values[spec['name']]
                if list(a.shape)!=spec['shape'] or a.dtype!=np.float16 or not np.isfinite(a).all():raise ValueError('Source interface mismatch: '+name)
                path=dest/(spec['name']+'.bin');a.astype('<f2',copy=False).tofile(path)
                if path.stat().st_size!=spec['bytes']:raise ValueError('Unexpected input byte count')
                inputs.append(dict(spec,file=str(path.resolve())))
        record=dict(scene=scene,sample=name,split='test' if name=='smoke_inputs.npz' else 'train',inputs=inputs)
        (dest/'metadata.json').write_text(json.dumps(record,indent=2));records.append(record)
    for spec in contracts:
        paths=[next(i['file'] for i in r['inputs'] if i['name']==spec['name']) for r in records if r['split']=='train']
        (folder/('calibration_'+spec['name']+'.txt')).write_text('\n'.join(paths)+'\n')
    (folder/'INPUTS.json').write_text(json.dumps(records,indent=2));return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES+['all'],default='all');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=[]
    for scene in SCENES if a.scene=='all' else [a.scene]:result.extend(run(scene,a.out))
    print(json.dumps(dict(prepared_samples=len(result),scenes=7 if a.scene=='all' else 1,precision='little-endian FP16',order=['raw_stack','reference_thumb'])))
