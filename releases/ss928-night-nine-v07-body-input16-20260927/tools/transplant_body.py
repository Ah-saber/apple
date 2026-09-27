"""Replace only a verified Conv16/ReLU body in an existing native ONNX.
No SDK compatibility or hardware latency is inferred from this rewrite.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper

def attr(node,key,default=None):
    return next((helper.get_attribute_value(a) for a in node.attribute if a.name==key),default)

def arrays(model):return {i.name:numpy_helper.to_array(i) for i in model.graph.initializer}

def convolution(node,weights):
    return (node.op_type=='Conv' and len(node.input)==3 and node.input[1] in weights and node.input[2] in weights
            and weights[node.input[1]].shape==(16,16,3,3) and weights[node.input[2]].shape==(16,)
            and attr(node,'group',1)==1 and list(attr(node,'pads',[0]*4))==[1]*4
            and list(attr(node,'strides',[1,1]))==[1,1] and list(attr(node,'dilations',[1,1]))==[1,1])

def follow_chain(model,convs):
    consumers={}
    for n in model.graph.node:
        for name in n.input:consumers.setdefault(name,[]).append(n)
    chain=[]
    for i,conv in enumerate(convs):
        following=consumers.get(conv.output[0],[])
        if len(following)!=1 or following[0].op_type!='Relu':raise ValueError('Body convolution must feed exactly one ReLU')
        activation=following[0]
        if i+1<len(convs):
            downstream=consumers.get(activation.output[0],[])
            if len(downstream)!=1 or downstream[0].name!=convs[i+1].name or convs[i+1].input[0]!=activation.output[0]:raise ValueError('Body must be an exclusive serial chain')
        chain.extend((conv,activation))
    if not chain:raise ValueError('Empty body')
    return chain

def source_chain(model,depths):
    weights=arrays(model);convs=[n for n in model.graph.node if '/body/' in n.name and n.op_type=='Conv']
    if len(convs) not in depths or not all(convolution(n,weights) for n in convs):raise ValueError('Unsupported source body geometry')
    return follow_chain(model,convs)

def transplant(native,reference,candidate):
    result=copy.deepcopy(native);nw=arrays(result);rw=arrays(reference);sw=arrays(candidate)
    frozen=source_chain(reference,(4,));student=source_chain(candidate,(1,2,3));matched=[]
    for old in frozen[::2]:
        found=[n for n in result.graph.node if convolution(n,nw)
               and np.array_equal(nw[n.input[1]].astype(np.float16),rw[old.input[1]].astype(np.float16))
               and np.array_equal(nw[n.input[2]].astype(np.float16),rw[old.input[2]].astype(np.float16))]
        if len(found)!=1:raise ValueError('Frozen body coefficients do not uniquely match native graph')
        matched.append(found[0])
    original=follow_chain(result,matched);dtype=nw[matched[0].input[1]].dtype
    if any(not n.name for n in original) or len({n.name for n in original})!=len(original):raise ValueError('Native body nodes require unique nonempty names')
    if any(sum(other.name==n.name for other in result.graph.node)!=1 for n in original):raise ValueError('Native body node name collides with another node')
    if not all(nw[n.input[1]].dtype==dtype and nw[n.input[2]].dtype==dtype for n in matched):raise ValueError('Native body uses mixed coefficient types')
    old_names={n.name for n in original};prefix='rawir_distilled_body/'
    if any(n.name.startswith(prefix) for n in result.graph.node) or any(i.name.startswith(prefix) for i in result.graph.initializer):raise ValueError('Body already rewritten')
    remap={student[0].input[0]:original[0].input[0]};new_nodes=[];new_initializers=[]
    for index,node in enumerate(student):
        new=copy.deepcopy(node);new.name=prefix+str(index)
        for i,name in enumerate(node.input):
            if name in sw:
                dest=prefix+name.replace('/','_');remap[name]=dest
                # Native coefficient dtype retained; conversion can alter source
                # arithmetic relative to Half execution, so it is recorded below.
                new_initializers.append(numpy_helper.from_array(sw[name].astype(dtype),dest))
            if name not in remap:raise ValueError('Candidate body has an external dependency')
            new.input[i]=remap[name]
        for i,name in enumerate(node.output):
            remap[name]=original[-1].output[0] if name==student[-1].output[0] else prefix+'value_'+str(index)+'_'+str(i);new.output[i]=remap[name]
        new_nodes.append(new)
    insert=min(i for i,n in enumerate(result.graph.node) if n.name in old_names)
    retained=[n for n in result.graph.node if n.name not in old_names];retained[insert:insert]=new_nodes
    del result.graph.node[:];result.graph.node.extend(retained);result.graph.initializer.extend(new_initializers)
    used={name for n in result.graph.node for name in n.input};keep=[i for i in result.graph.initializer if i.name in used]
    del result.graph.initializer[:];result.graph.initializer.extend(keep)
    onnx.checker.check_model(result)
    report={'old_body_depth':4,'new_body_depth':len(student)//2,'native_coefficient_dtype':str(dtype),
            'source_coefficient_dtype':str(sw[student[0].input[1]].dtype),'match_precision':'float16 coefficient equality',
            'body_only_rewrite':True,'SDK_verified':False,'NPU_timing_verified':False,
            'source_arithmetic_identical':False,'note':'Coefficient dtype and SDK activation/calibration require deployment-side validation.'}
    return result,report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    result,report=transplant(onnx.load(a.native),onnx.load(a.reference),onnx.load(a.candidate));onnx.save(result,dest);dest.with_suffix('.rewrite.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
