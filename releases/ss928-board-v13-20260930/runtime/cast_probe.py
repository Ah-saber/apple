"""Isolate the SDK's actual direct-Cast rounding and output Report mapping."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


class DirectCast(torch.nn.Module):
    def forward(self, value):
        return value.to(torch.uint8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--onnx-site',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.onnx_site));import onnx
    a.out.mkdir(parents=True,exist_ok=True)
    values=np.array([0,.25,.49,.5,.51,.75,1,1.25,1.49,1.5,1.51,1.75,
                     2.49,2.5,2.51,3.5,47.49,47.5,47.51,48.5,127.5,128.5,
                     229.49,229.5,229.51,254.49,254.5,254.51,254.75,255,10.5,11.5],dtype=np.float32)
    matrix=np.tile(values,2).reshape(1,1,8,8)
    for dtype,label in [(torch.float32,'fp32'),(torch.float16,'fp16')]:
        x=torch.from_numpy(matrix).to(dtype)
        path=a.out/f'direct_cast_{label}.onnx'
        torch.onnx.export(DirectCast(),x,str(path),input_names=['float_gray'],output_names=['gray_u8'],
                          opset_version=17,dynamo=False)
        onnx.checker.check_model(str(path))
        np.savez_compressed(a.out/f'direct_cast_{label}.npz',float_gray=x.numpy(),
                            expected_trunc=DirectCast()(x).numpy(),expected_round_even=np.rint(x.numpy()).astype(np.uint8))
    (a.out/'README.json').write_text(json.dumps({'shape':[1,1,8,8],'values':values.tolist(),
        'purpose':'Run in isolation before selecting UINT8 full graphs; actual SDK Cast rounding is unknown.',
        'model_time_boundary':'SDK model invocation, including its output Report'},indent=2))


if __name__=='__main__':main()
