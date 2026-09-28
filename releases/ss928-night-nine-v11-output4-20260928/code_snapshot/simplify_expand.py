"""Fold static output Expand shape generation into one constant."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

def rewrite(source,target):
    graph=onnx.load(str(source));nodes=list(graph.graph.node);by_output={x:n for n in nodes for x in n.output}
    expands=[n for n in nodes if n.name=='/output/Expand'];assert len(expands)==1
    expand=expands[0];where=by_output[expand.input[1]];assert where.op_type=='Where'
    equal=by_output[where.input[0]];assert equal.op_type=='Equal'
    constant_shape=by_output[where.input[2]];assert constant_shape.op_type=='Constant'
    shape=numpy_helper.to_array(constant_shape.attribute[0].t)
    assert list(shape)==[1,1,64 if 'small' in source.name else 1024,3,96 if 'small' in source.name else 1280,3],shape
    mul=by_output[equal.input[1]];assert mul.op_type=='Mul'
    zero_shape=by_output[mul.input[0]];assert zero_shape.op_type=='ConstantOfShape'
    chain=[where,equal,mul,zero_shape,by_output[zero_shape.input[0]],by_output[mul.input[1]],by_output[equal.input[0]]]
    # The other Where input is the same ConstantOfShape tensor.
    assert where.input[1]==zero_shape.output[0]
    for n in chain:
        assert len(n.output)==1
        uses=sum(n.output[0] in m.input for m in nodes)
        assert uses>=1,(n.name,uses)
    expand.input[1]=constant_shape.output[0]
    for n in chain:graph.graph.node.remove(n)
    assert not any(n.name in ('/output/Where','/output/Equal','/output/ConstantOfShape') for n in graph.graph.node)
    onnx.checker.check_model(graph);onnx.save(graph,str(target))
    return {'source':source.name,'target':target.name,'removed_output_shape_nodes':len(chain),'expand_shape':shape.tolist()}

def verify(source,target):
    options=ort.SessionOptions();options.intra_op_num_threads=2
    a=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    b=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(928);feed={m.name:rng.uniform(.05,.95,[v if isinstance(v,int) else 1 for v in m.shape]).astype(np.float16 if m.type=='tensor(float16)' else np.float32) for m in a.get_inputs()}
    x=a.run(None,feed)[0];y=b.run(None,feed)[0]
    return {'exact':bool(np.array_equal(x,y)),'different_pixels':int(np.count_nonzero(x!=y)),'shape':list(y.shape)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--stems',nargs='+',required=True);a=p.parse_args();rows=[]
    for stem in a.stems:
        for small in (False,True):
            suffix='_small' if small else ''
            source=a.directory/f'{stem}{suffix}.onnx';target=a.directory/f'{stem}_static{suffix}.onnx'
            row=rewrite(source,target)
            if small:
                row.update(verify(source,target));assert row['exact'],row
            rows.append(row);print(row,flush=True)
    (a.directory/'expand_static_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
