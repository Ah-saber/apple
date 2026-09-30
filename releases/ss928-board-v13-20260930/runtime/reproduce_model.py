"""Rebuild this release's five-scene models using only packaged source and weights."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--scene',choices=['day_normal','weather_light','weather_medium','weather_heavy','weather_heavy_c32'],required=True)
    p.add_argument('--kind',choices=['equivalent','fused','temporal','current'],required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');a=p.parse_args()
    sys.path[:0]=[str(a.release/'source'),str(a.release/'runtime/legacy')]
    from ir_sr.model import GlobalReference
    from half_student import make_student
    from train_quarter_student import BoxQuarter
    from compact_model import CompactModel, BoardCompact
    from equivalent_model import BoardHalfOptimized
    if not torch.cuda.is_available():raise RuntimeError('Use the test environment with CUDA; frozen sampling coordinates were verified on GPU')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=json.loads((a.release/'records/deployment'/a.scene/'config.json').read_text())
    metadata=json.loads((a.release/'records/deployment'/a.scene/'manifest.json').read_text())
    group='day' if a.scene=='day_normal' else ('light' if a.scene in ['weather_light','weather_medium'] else 'heavy')
    if a.kind=='current' and group!='day':raise ValueError('Current-only folding belongs to the preserved day alternative')
    if a.kind in ['fused','temporal'] and group=='day':raise ValueError('The two day compact candidates were rejected')
    if a.kind in ['equivalent','current']:
        name='half-day-rawskip' if a.kind=='current' else {'day':'half-day-nine-temporal','light':'half-light-temporal','heavy':'half-heavy-temporal'}[group]
        checkpoint=a.release/'models/baseline'/(name+'.pt')
    else:checkpoint=a.release/'models'/(group+'_'+a.kind+'.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    reference=GlobalReference(16,config.get('absolute_raw_reference',False),
                              config.get('reference_pyramid',False),config.get('reference_sensor_y',False))
    box=torch.tensor([metadata['context_box']],device='cuda')
    if a.kind in ['equivalent','current']:
        source=BoxQuarter(make_student(reference,state),current_only=state.get('current_only',False))
        source.load_state_dict(state['model'],strict=True)
        model=BoardHalfOptimized(source,box,single_frame=a.kind=='current')
    else:
        spec=state['spec']
        source=CompactModel(reference,spec['width'],spec['depth'],spec['raw_skip'],spec['late_reference'])
        source.load_state_dict(state['model'],strict=True)
        model=BoardCompact(source,box)
    dtype=torch.float32 if a.precision=='fp32' else torch.float16
    model=model.cuda().to(dtype=dtype).eval()
    sample=np.load(a.sample,allow_pickle=False)
    x=torch.from_numpy(sample['nine_raw']).cuda().to(dtype)
    if a.kind=='current':x=x[:,-1:]
    context=torch.from_numpy(sample['reference_thumb']).cuda().to(dtype)
    with torch.inference_mode():output=model(x,context).float().cpu().numpy()
    assert tuple(output.shape)==(1,1,3072,3840)
    np.savez_compressed(a.out,display_gray=output)
    record={'checkpoint':str(checkpoint),'precision':a.precision,'shape':list(output.shape),'sample':str(a.sample)}
    if a.kind=='equivalent' and a.precision=='fp32':
        d=output[:,:,::3,::3]-sample['optimized_native_gray']
        record.update({'mae_gray':float(np.abs(d).mean()),'max_gray':float(np.abs(d).max())})
        assert record['max_gray']<.01,record
    a.out.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
