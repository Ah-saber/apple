"""Compare a dequantized complete board output against the exact delivered ONNX PC output."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--role',choices=['retained_original','tied500'],required=True)
    p.add_argument('--board-output',type=Path,required=True);p.add_argument('--dump-dtype',choices=['FP16','FP32'],default='FP16');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    folder=ROOT/'models'/a.scene;meta=json.loads((folder/'manifest.json').read_text());contract=meta['graphs'][a.role];spec=contract['outputs'][0]
    dtype='<f2' if a.dump_dtype=='FP16' else '<f4';expected_bytes=int(np.prod(spec['shape']))*np.dtype(dtype).itemsize
    if a.board_output.stat().st_size!=expected_bytes:raise ValueError('Raw dump byte count mismatch; remove SDK headers or specify the actual dequantized dump type')
    board=np.fromfile(a.board_output,dtype=dtype).reshape(spec['shape']).astype(np.float32)
    with np.load(folder/contract['PC_output']) as saved:pc=saved[spec['name']].astype(np.float32)
    if not np.isfinite(board).all():raise ValueError('Non-finite board output')
    error=board-pc;absolute=np.abs(error)
    result=dict(scene=a.scene,role=a.role,source_model=contract['file'],source_input='smoke_inputs.npz',frame=37,PC_backend=contract['PC_backend'],
        output_shape=spec['shape'],graph_output_dtype=spec['dtype'],board_dump_dtype=a.dump_dtype,bytes=expected_bytes,
        MAE_gray=float(absolute.mean()),max_abs_gray=float(absolute.max()),P95_abs_gray=float(np.percentile(absolute,95)),mean_bias_gray=float(error.mean()),RMSE_gray=float(np.sqrt(np.mean(error**2))),
        board_min_gray=float(board.min()),board_max_gray=float(board.max()),PC_min_gray=float(pc.min()),PC_max_gray=float(pc.max()),
        GT_full_frame=True,metric_roi_tlhw=meta['smoke']['metric_roi_tlhw'],quality_or_latency_pass_not_inferred=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
