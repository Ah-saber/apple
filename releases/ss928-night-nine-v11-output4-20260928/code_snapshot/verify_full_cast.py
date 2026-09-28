"""Compare complete 3x AddHalf/Cast graph with source Round output on real frames."""
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--vectors',type=Path,required=True)
    a=p.parse_args();options=ort.SessionOptions();options.intra_op_num_threads=2
    rows=[]
    for scene,frame in (('ordinary',20),('special',60)):
        source=ort.InferenceSession(str(a.source_dir/f'{scene}_rows32.onnx'),options,providers=['CPUExecutionProvider'])
        candidate=ort.InferenceSession(str(a.graphs/f'{scene}_rows32_addhalf_cast_u8.onnx'),options,providers=['CPUExecutionProvider'])
        vector=np.load(a.vectors/f'{scene}_frame_{frame}.npz')
        feed={meta.name:vector[meta.name].astype(np.float16 if meta.type=='tensor(float16)' else np.float32) for meta in source.get_inputs()}
        gray=source.run(None,feed)[0];actual=candidate.run(None,feed)[0]
        halfup=np.trunc((gray+np.float16(.5)).astype(np.float32)).astype(np.uint8)
        even=np.rint(gray.astype(np.float32)).astype(np.uint8)
        diff=actual.astype(np.int16)-even.astype(np.int16)
        row={'scene':scene,'frame':frame,'shape':list(actual.shape),'exact_half_up':bool(np.array_equal(actual,halfup)),'different_from_even_pixels':int(np.count_nonzero(diff)),'fraction_different_from_even':float(np.mean(diff!=0)),'mean_bias_gray':float(diff.mean()),'max_abs_gray':int(np.abs(diff).max())}
        assert row['exact_half_up'] and row['max_abs_gray']<=1,row
        rows.append(row);print(row,flush=True)
    (a.graphs.parent/'full_cast_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
