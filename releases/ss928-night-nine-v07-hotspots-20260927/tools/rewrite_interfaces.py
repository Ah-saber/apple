"""One-factor interface/pointwise-scale controls; retain all model computation."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper

def change(model,mode):
    model=copy.deepcopy(model);prefix='v07_interface__'
    if any(v.startswith(prefix) for n in model.graph.node for v in list(n.input)+list(n.output)):raise ValueError('Namespace exists')
    if mode=='input_nchw_half':
        inputs=[v for v in model.graph.input if [d.dim_value for d in v.type.tensor_type.shape.dim]==[1,9,1024,1280]]
        if len(inputs)!=1 or inputs[0].type.tensor_type.elem_type!=TensorProto.FLOAT:raise ValueError('Require one full NCHW FP32 nine-frame input')
        raw=inputs[0];source=raw.name;cast=prefix+'canonical_float'
        for n in model.graph.node:
            for i,v in enumerate(n.input):
                if v==source:n.input[i]=cast
        model.graph.node.insert(0,helper.make_node('Cast',[source],[cast],to=TensorProto.FLOAT))
        raw.type.tensor_type.elem_type=TensorProto.FLOAT16
    elif mode=='output_half':
        if len(model.graph.output)!=1 or model.graph.output[0].type.tensor_type.elem_type!=TensorProto.FLOAT:raise ValueError('Require one FP32 output')
        old=model.graph.output[0].name;new=prefix+'full_float';count=0
        for n in model.graph.node:
            for i,v in enumerate(n.output):
                if v==old:n.output[i]=new;count+=1
            for i,v in enumerate(n.input):
                if v==old:n.input[i]=new
        if count!=1:raise ValueError('Require unique output producer')
        model.graph.node.append(helper.make_node('Cast',[new],[old],to=TensorProto.FLOAT16));model.graph.output[0].type.tensor_type.elem_type=TensorProto.FLOAT16
    elif mode=='scale_packed_f32':
        matches=[n for n in model.graph.node if n.op_type=='DepthToSpace' and any(a.name=='blocksize' and a.i==6 for a in n.attribute)]
        if len(matches)!=1 or len(model.graph.output)!=1:raise ValueError('Require unique shuffle6/output')
        old=matches[0];producers={o:n for n in model.graph.node for o in n.output};const={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
        for n in model.graph.node:
            if n.op_type=='Constant':
                for a in n.attribute:
                    if a.name=='value':const[n.output[0]]=numpy_helper.to_array(a.t)
        value=model.graph.output[0].name;chain=[]
        while value!=old.output[0]:
            n=producers.get(value)
            if n is None or n.op_type not in ('Cast','Clip','Mul','Identity'):raise ValueError('Unsupported post-shuffle path')
            operands=[i for i in n.input if i and i not in const]
            if len(operands)!=1:raise ValueError('Only scalar pointwise operands supported')
            for i in n.input:
                if i in const and const[i].size!=1:raise ValueError('Nonscalar constant')
            chain.append((n,operands[0]));value=operands[0]
        if not chain:raise ValueError('No pointwise scaling path')
        # Never clone this path if its intermediate values feed another branch.
        chain_nodes=[n for n,_ in chain];produced={v for n,_ in chain for v in n.output}
        for n in model.graph.node:
            if n not in chain_nodes and any(i in produced for i in n.input):raise ValueError('Output chain has external consumer')
        value=old.input[0];extra=[]
        for j,(node,operand) in enumerate(reversed(chain)):
            new=copy.deepcopy(node);new.name=prefix+f'pointwise_{j}'
            for i,v in enumerate(new.input):
                if v==operand:new.input[i]=value
                elif v in const:
                    label=prefix+f'constant_{j}_{i}';model.graph.initializer.append(numpy_helper.from_array(const[v],label));new.input[i]=label
            if len(new.output)!=1:raise ValueError('Require single pointwise output')
            value=prefix+f'value_{j}';new.output[0]=value;extra.append(new)
        old.input[0]=value;old.output[0]=model.graph.output[0].name
        nodes=[]
        for n in model.graph.node:
            if n in chain_nodes:continue
            if n==old:nodes.extend(extra)
            nodes.append(n)
        del model.graph.node[:];model.graph.node.extend(nodes)
    else:raise ValueError(mode)
    del model.graph.value_info[:];onnx.checker.check_model(model);return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',required=True,choices=['input_nchw_half','output_half','scale_packed_f32']);a=p.parse_args();path=Path(a.output)
    if path.exists() or path.with_suffix('.rewrite.json').exists():raise FileExistsError(path)
    model=change(onnx.load(a.input),a.mode);path.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,path);path.with_suffix('.rewrite.json').write_text(json.dumps({'source':a.input,'one_factor':a.mode,'SDK_compiled':False,'NPU_measured':False},indent=2))
if __name__=='__main__':main()
