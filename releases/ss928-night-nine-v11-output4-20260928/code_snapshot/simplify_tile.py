"""Remove the identity Expand emitted before output Tile in fixed-shape graphs."""
import argparse, json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

def const_value(node):
    assert node.op_type=='Constant'
    return numpy_helper.to_array(next(a.t for a in node.attribute if a.name=='value'))

def rewrite(source,target):
    graph=onnx.load(str(source));by_output={out:n for n in graph.graph.node for out in n.output}
    tiles=[n for n in graph.graph.node if n.op_type=='Tile' and n.name=='/output/Tile']
    assert len(tiles)==1
    tile=tiles[0];expand=by_output[tile.input[0]];assert expand.op_type=='Expand'
    shape=by_output[expand.input[1]];assert shape.op_type=='ConstantOfShape'
    count=by_output[shape.input[0]];assert count.op_type=='Constant' and np.array_equal(const_value(count),np.array([6]))
    assert len(shape.attribute)==1 and np.array_equal(numpy_helper.to_array(shape.attribute[0].t),np.array([1]))
    for name in (expand.output[0],shape.output[0],count.output[0]):
        uses=sum(name in n.input for n in graph.graph.node)
        assert uses==1,(name,uses)
    tile.input[0]=expand.input[0]
    for node in (expand,shape,count):graph.graph.node.remove(node)
    assert not any(n.op_type=='Expand' and n.name.startswith('/output/') for n in graph.graph.node)
    onnx.checker.check_model(graph);onnx.save(graph,str(target))
    return {'source':source.name,'target':target.name,'nodes_removed':3,'remaining_output_expand':0}

def verify(source,target):
    options=ort.SessionOptions();options.intra_op_num_threads=2
    old=ort.InferenceSession(str(source),options,providers=['CPUExecutionProvider'])
    new=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(928);feed={}
    for m in old.get_inputs():
        feed[m.name]=rng.uniform(.05,.95,[v if isinstance(v,int) else 1 for v in m.shape]).astype(np.float16 if m.type=='tensor(float16)' else np.float32)
    a=old.run(None,feed)[0];b=new.run(None,feed)[0]
    return {'exact':bool(np.array_equal(a,b)),'different_pixels':int(np.count_nonzero(a!=b)),'shape':list(b.shape)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--stems',nargs='+',required=True);a=p.parse_args();rows=[]
    for stem in a.stems:
        for small in (False,True):
            suffix='_small' if small else ''
            source=a.directory/f'{stem}{suffix}.onnx'
            target=a.directory/f'{stem}_static{suffix}.onnx'
            row=rewrite(source,target)
            if small:
                row.update(verify(source,target));assert row['exact'],row
            rows.append(row);print(row,flush=True)
    (a.directory/'tile_static_verification.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
