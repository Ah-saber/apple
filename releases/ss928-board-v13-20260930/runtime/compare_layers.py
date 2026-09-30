"""Compare corresponding dequantized board tensors with PC stage references."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pc',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    pc=np.load(a.pc,allow_pickle=False); board=np.load(a.board,allow_pickle=False)
    names=[k for k in pc.files if k in board.files]
    if not names: raise ValueError('No matching stages; retain PC stage names in the dequantized board dump')
    report={'pc':str(a.pc),'board':str(a.board),'stages':{},'missing_board_stages':[k for k in pc.files if k not in board.files]}
    for name in names:
        x=pc[name].astype(np.float64); y=board[name]
        if y.dtype.kind not in 'fc':
            raise ValueError(f'{name}: integer dump requires actual SDK scales and dequantization first')
        y=y.astype(np.float64)
        if x.shape!=y.shape: raise ValueError(f'{name}: {x.shape} vs {y.shape}; do not guess layout')
        if not np.isfinite(y).all(): raise ValueError(f'{name}: nonfinite board values')
        d=y-x
        item={'shape':list(x.shape),'mae':float(np.abs(d).mean()),'signed_bias':float(d.mean()),
              'max_abs':float(np.abs(d).max()),'rmse':float(np.sqrt((d*d).mean()))}
        if x.ndim==4:
            item['channel_signed_bias']=d.mean((0,2,3)).tolist()
            if x.shape[1]==4: item['phase_bias_range']=float(np.ptp(d.mean((0,2,3))))
        report['stages'][name]=item
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
