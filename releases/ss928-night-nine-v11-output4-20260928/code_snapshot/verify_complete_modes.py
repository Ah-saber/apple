"""Real-frame complete-output source checks for selected FP16 and UINT8 modes."""
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort

def main():
    p=argparse.ArgumentParser();p.add_argument('--graphs',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--stems',nargs='+',required=True);a=p.parse_args()
    options=ort.SessionOptions();options.intra_op_num_threads=2;rows=[]
    for stem in a.stems:
        scene=stem.split('_',1)[0];frame=20 if scene=='ordinary' else 60
        base=ort.InferenceSession(str(a.graphs/f'{stem}.onnx'),options,providers=['CPUExecutionProvider'])
        vector=np.load(a.vectors/f'{scene}_frame_{frame}.npz')
        feed={m.name:vector[m.name].astype(np.float16 if m.type=='tensor(float16)' else np.float32) for m in base.get_inputs()}
        gray=base.run(None,feed)[0]
        even=np.rint(gray.astype(np.float32)).astype(np.uint8)
        for mode in ('quantizelinear','addhalf_cast'):
            session=ort.InferenceSession(str(a.graphs/f'{stem}_{mode}_u8.onnx'),options,providers=['CPUExecutionProvider'])
            actual=session.run(None,feed)[0]
            expected=even if mode=='quantizelinear' else np.trunc((gray+np.float16(.5)).astype(np.float32)).astype(np.uint8)
            diff=actual.astype(np.int16)-even.astype(np.int16)
            mode_diff=actual.astype(np.int16)-expected.astype(np.int16)
            row={'scene':scene,'stem':stem,'mode':mode,'frame':frame,'shape':list(actual.shape),'exact_same_backend':bool(np.array_equal(actual,expected)),'different_from_mode_expected_pixels':int(np.count_nonzero(mode_diff)),'mode_max_abs_gray':int(np.abs(mode_diff).max()),'different_from_even_pixels':int(np.count_nonzero(diff)),'fraction_different_from_even':float(np.mean(diff!=0)),'mean_bias_gray':float(diff.mean()),'max_abs_gray':int(np.abs(diff).max())}
            if mode=='quantizelinear':assert row['exact_same_backend'],row
            else:assert row['mode_max_abs_gray']<=1,row
            rows.append(row);print(row,flush=True)
    (a.graphs.parent/'complete_output_mode_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
