"""Unpack source arrays into exact model-input files and PC display references."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=['fp32','fp16'])
    p.add_argument('--model',type=Path,help='Use each ONNX input type, including mixed FP16/FP32 night inputs')
    p.add_argument('--inputs-only',action='store_true')
    p.add_argument('--current-only',action='store_true')
    p.add_argument('--reference',choices=['native','display'],default='native')
    a=p.parse_args()
    if not a.model and not a.precision:p.error('Specify --model or --precision')
    data=np.load(a.input,allow_pickle=False)
    dtypes={}
    if a.model:
        import onnx
        graph=onnx.load(str(a.model))
        for value in graph.graph.input:
            dtypes[value.name]=np.dtype({1:'<f4',10:'<f2'}[value.type.tensor_type.elem_type])
        if 'current_raw' in dtypes:a.current_only=True
    if a.current_only and not a.inputs_only and 'pc_current_native_gray' not in data.files:
        raise ValueError('This sample lacks the folded-current model reference; use --inputs-only and regenerate_reference.py with the exact graph')
    a.out.mkdir(parents=True,exist_ok=False)
    x=data['nine_raw']; c=data['reference_thumb']
    if a.current_only: x=x[:,-1:]
    for name,value in [('current_raw' if a.current_only else 'nine_raw',x),('reference_thumb',c)]:
        if not np.isfinite(value).all():raise ValueError(name+' contains nonfinite values')
        dtype=dtypes.get(name,np.dtype('<f4' if a.precision=='fp32' else '<f2'))
        np.ascontiguousarray(value,dtype=dtype).tofile(a.out/(name+'.bin'))
        dtypes[name]=dtype
    expected=None
    if not a.inputs_only:
        expected=data['pc_current_native_gray'] if a.current_only else data['pc_native_gray']
        if a.reference=='display': expected=np.repeat(np.repeat(expected,3,axis=-2),3,axis=-1)
        np.save(a.out/'pc_reference.npy',expected,allow_pickle=False)
    metadata={'source':str(a.input),'input_dtype':{k:str(v) for k,v in dtypes.items()},'input_layout':'NCHW',
              'nine_raw_shape':list(x.shape),'reference_thumb_shape':list(c.shape),
              'pc_reference_shape':None if expected is None else list(expected.shape),
              'PC_reference_precision':'See source manifest; casting inputs does not regenerate a model reference.'}
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':main()
