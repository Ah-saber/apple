"""Interpolate matching half-grid checkpoints; select using validation only."""
import argparse
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--temporal',type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert 0<=a.alpha<=1
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    temporal=torch.load(a.temporal,map_location='cpu',weights_only=False)
    assert base['model'].keys()==temporal['model'].keys()
    assert base.get('grid')==temporal.get('grid')=='half'
    state={}
    for key,value in base['model'].items():
        other=temporal['model'][key]
        assert value.shape==other.shape
        if value.is_floating_point():
            state[key]=torch.lerp(value.float(),other.float(),a.alpha).to(value.dtype)
        else:
            assert torch.equal(value,other)
            state[key]=value
    result={key:base[key] for key in ('format','scenes','depth','front_kind',
             'grid','width','raw_skip','current_only') if key in base}
    result.update({'model':state,'base_checkpoint':str(a.base),
        'step':temporal.get('step',base.get('step',0)),
        'temporal_checkpoint':str(a.temporal),'temporal_alpha':a.alpha,
        'selection':'validation temporal metrics pending',
        'test_used_for_selection':False})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,a.out)
    print('saved',a.out,'alpha',a.alpha,flush=True)


if __name__=='__main__':main()
