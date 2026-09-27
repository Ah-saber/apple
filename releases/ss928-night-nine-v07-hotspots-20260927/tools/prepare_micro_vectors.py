"""Generate full-size micro inputs from real canonical vectors and same PC graph.
Requires deployment PC's existing ONNX Runtime, never changes model boundary.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--vector',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--engine',choices=['ort','reference'],default='ort');a=p.parse_args()
    dest=Path(a.output_dir)
    if dest.exists():raise FileExistsError(dest)
    model=onnx.load(a.model);shuffles=[n for n in model.graph.node if n.op_type=='DepthToSpace' and any(attr.name=='blocksize' and attr.i==6 for attr in n.attribute)]
    if len(shuffles)!=1:raise ValueError('Require original front_f32 shuffle6 graph')
    inferred=onnx.shape_inference.infer_shapes(model);meta={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)};name=shuffles[0].input[0]
    if name not in meta:raise ValueError('Missing packed type/shape')
    packed_info=copy.deepcopy(meta[name]);dims=packed_info.type.tensor_type.shape.dim
    if [d.dim_value for d in dims][1:]!=[36,512,640]:raise ValueError('Expected complete packed tensor')
    if len(dims)!=4 or dims[0].dim_value not in (0,1):raise ValueError('Batch must be one')
    dims[0].ClearField('dim_param');dims[0].dim_value=1
    producers={o:n for n in model.graph.node for o in n.output};live=set()
    def visit(v):
        if not v or v in live:return
        live.add(v)
        if v in producers:
            for i in producers[v].input:visit(i)
    visit(name)
    kept=[n for n in model.graph.node if any(o in live for o in n.output)];initializers=[v for v in model.graph.initializer if v.name in live]
    del model.graph.node[:];model.graph.node.extend(kept);del model.graph.initializer[:];model.graph.initializer.extend(initializers);del model.graph.output[:];model.graph.output.append(packed_info);del model.graph.value_info[:]
    inputs=[v for v in model.graph.input if v.name in live];del model.graph.input[:];model.graph.input.extend(inputs);onnx.checker.check_model(model)
    v=np.load(a.vector);raw=np.asarray(v['nine_raw']);thumb=np.asarray(v['reference_thumb']);feeds={}
    for inp in model.graph.input:
        if inp.name not in live:continue
        shape=[d.dim_value for d in inp.type.tensor_type.shape.dim];typ=inp.type.tensor_type.elem_type
        if typ not in (TensorProto.FLOAT,TensorProto.FLOAT16):raise ValueError('Unsupported input dtype')
        dtype=np.float16 if typ==TensorProto.FLOAT16 else np.float32
        if shape==[1,9,1024,1280]:value=raw
        elif shape==[1,1,64,64]:value=thumb
        else:raise ValueError('Require original NCHW front_f32 interface')
        feeds[inp.name]=np.ascontiguousarray(value,dtype=dtype)
    if a.engine=='ort':
        import onnxruntime as ort
        session=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
    else:
        from onnx.reference import ReferenceEvaluator
        session=ReferenceEvaluator(model)
    packed=session.run([name],feeds)[0]
    if packed.shape!=(1,36,512,640) or not np.isfinite(packed).all():raise ValueError('Invalid packed output')
    dest.mkdir(parents=True);np.savez_compressed(dest/'canonical_vectors.npz',nine_raw=raw,reference_thumb=thumb,packed=packed)
    records=[]
    for label,value in [('statistics_fp32',raw.astype(np.float32)),('statistics_fp16',raw.astype(np.float16)),('output_fp32',packed.astype(np.float32)),('output_fp16',packed.astype(np.float16))]:
        value=np.ascontiguousarray(value);file=dest/(label+'.bin');value.tofile(file);records.append({'file':file.name,'shape':list(value.shape),'dtype':str(value.dtype),'bytes':file.stat().st_size})
    # Independent expected output permutation, in normalized packed units, no gray scaling.
    expected=packed.reshape(1,1,6,6,512,640).transpose(0,1,4,2,5,3).reshape(1,1,3072,3840)
    np.savez_compressed(dest/'expected_output_permutation.npz',normalized_output=expected)
    (dest/'inputs.json').write_text(json.dumps({'whole_source_model':a.model,'canonical_vector':a.vector,'packed_reference_units':'normalized; no 255 scaling','files':records,'micro_only':True,'full_model_speed_claim':False},indent=2))
if __name__=='__main__':main()
