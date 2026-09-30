"""Use the exact ONNX graph and its declared input types for a PC reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=np.load(a.sample,allow_pickle=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(a.model),sess_options=options,providers=['CPUExecutionProvider'])
    feed={}
    for value in session.get_inputs():
        key='nine_raw' if value.name=='current_raw' else value.name
        x=data[key]
        if value.name=='current_raw': x=x[:,-1:]
        dtype={'tensor(float)':np.float32,'tensor(float16)':np.float16}[value.type]
        feed[value.name]=np.ascontiguousarray(x,dtype=dtype)
    output=session.run(None,feed)
    np.savez_compressed(a.out,**{v.name:x for v,x in zip(session.get_outputs(),output)})
    metadata={'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
              'sample':str(a.sample),'provider':'CPUExecutionProvider','onnxruntime':ort.__version__,
              'inputs':{k:{'shape':list(x.shape),'dtype':str(x.dtype)} for k,x in feed.items()},
              'outputs':{v.name:{'shape':list(x.shape),'dtype':str(x.dtype)} for v,x in zip(session.get_outputs(),output)}}
    a.out.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':main()
