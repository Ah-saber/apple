"""Add two complete-model UINT8 output modes to selected FP16 graphs."""
import argparse, json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

def add_mode(source, target, mode):
    graph=onnx.load(str(source));output=graph.graph.output[0];old=output.name
    if mode=='quantizelinear':
        converted='display_gray_float32_for_quantize';scale='display_gray_scale';zero='display_gray_zero'
        graph.graph.node.append(helper.make_node('Cast',[old],[converted],name='output/CastFP32',to=TensorProto.FLOAT))
        graph.graph.initializer.append(numpy_helper.from_array(np.array(1,dtype=np.float32),scale))
        graph.graph.initializer.append(numpy_helper.from_array(np.array(0,dtype=np.uint8),zero))
        new='display_gray_quantized'
        graph.graph.node.append(helper.make_node('QuantizeLinear',[converted,scale,zero],[new],name='output/QuantizeLinear'))
    elif mode=='addhalf_cast':
        scalar='display_gray_half_scalar';added='display_gray_add_half'
        graph.graph.initializer.append(numpy_helper.from_array(np.array(.5,dtype=np.float16),scalar))
        graph.graph.node.append(helper.make_node('Add',[old,scalar],[added],name='output/AddHalf'))
        new='display_gray_cast_u8'
        graph.graph.node.append(helper.make_node('Cast',[added],[new],name='output/CastU8',to=TensorProto.UINT8))
    else:raise ValueError(mode)
    output.name=new;output.type.tensor_type.elem_type=TensorProto.UINT8
    onnx.checker.check_model(graph);onnx.save(graph,str(target))

def verify(source, candidate, mode):
    options=ort.SessionOptions();options.intra_op_num_threads=2
    old=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    new=ort.InferenceSession(str(candidate),options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(928);feed={}
    for meta in old.get_inputs():
        shape=[d if isinstance(d,int) else 1 for d in meta.shape]
        feed[meta.name]=rng.uniform(.05,.95,shape).astype(np.float16 if meta.type=='tensor(float16)' else np.float32)
    gray=old.run(None,feed)[0];actual=new.run(None,feed)[0]
    expected=np.rint(np.clip(gray.astype(np.float32),0,255)).astype(np.uint8) if mode=='quantizelinear' else np.trunc((gray+np.float16(.5)).astype(np.float32)).astype(np.uint8)
    difference=actual.astype(np.int16)-expected.astype(np.int16)
    item={'source':source.name,'candidate':candidate.name,'mode':mode,
          'exact_same_backend':bool(np.array_equal(actual,expected)),
          'different_pixels':int(np.count_nonzero(difference)),
          'max_abs_gray':int(np.abs(difference).max()),
          'mean_signed_gray':float(difference.mean()),'output_shape':list(actual.shape)}
    if mode=='quantizelinear':assert item['exact_same_backend'],item
    else:assert item['max_abs_gray']<=1,item
    return item

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--stems',nargs='+',required=True);a=p.parse_args();rows=[]
    for stem in a.stems:
        for small in (False,True):
            suffix='_small' if small else ''
            source=a.source_dir/f'{stem}{suffix}.onnx'
            for mode in ('quantizelinear','addhalf_cast'):
                target=a.source_dir/f'{stem}_{mode}_u8{suffix}.onnx'
                add_mode(source,target,mode)
                if small:
                    item=verify(source,target,mode);rows.append(item);print(item,flush=True)
    (a.source_dir/'output_modes_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
