"""Resolve only Identity aliases of initializers; preserve learned tensor bytes."""
import argparse,copy,hashlib,json
from pathlib import Path
import onnx

def materialize(model):
    model=copy.deepcopy(model);constants={t.name:t for t in model.graph.initializer};kept=[];aliases=[]
    for node in model.graph.node:
        if node.op_type=='Identity' and len(node.input)==1 and len(node.output)==1 and node.input[0] in constants:
            t=onnx.TensorProto();t.CopyFrom(constants[node.input[0]]);t.name=node.output[0]
            if t.name in constants:raise ValueError('Alias initializer collision')
            model.graph.initializer.append(t);constants[t.name]=t
            aliases.append({'node':node.name,'source':node.input[0],'target':node.output[0]})
        else:kept.append(node)
    del model.graph.node[:];model.graph.node.extend(kept);onnx.checker.check_model(model)
    return model,aliases

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m,aliases=materialize(onnx.load(str(a.input)));a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output))
    record={'materialized_initializer_aliases':aliases,'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'learned_values_unchanged':True,'NPU_compile_verified':False};a.output.with_suffix('.aliases.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
