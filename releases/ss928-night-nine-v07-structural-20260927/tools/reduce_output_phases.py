"""Approximate Conv36 output with fewer phases; retain complete image output.
Only the learned output path is replaced; native reference compatibility stays.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from materialize_aliases import materialize

def attributes(node):return {a.name:helper.get_attribute_value(a) for a in node.attribute}

def reduce_phases(model,factor,mode='nearest',projection_source=None,frozen_source=None):
    if factor not in (1,2,3) or mode not in ('nearest','bilinear','fixed','preserve_nearest','preserve_fixed'):raise ValueError('Invalid phase approximation')
    model,_=materialize(model);prefix='v07_reduced_phase__'
    if any(v.startswith(prefix) for n in model.graph.node for v in list(n.input)+list(n.output)):raise ValueError('Namespace already present')
    constants={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type=='Constant' and 'value' in attributes(n):constants[n.output[0]]=numpy_helper.to_array(attributes(n)['value'])
    matches=[n for n in model.graph.node if n.op_type=='DepthToSpace' and attributes(n).get('blocksize')==6]
    if len(matches)!=1:raise ValueError('Require unique shuffle6')
    shuffle=matches[0]
    if attributes(shuffle).get('mode',b'DCR')!=b'CRD':raise ValueError('Require CRD phase order')
    producers={v:n for n in model.graph.node for v in n.output};value=shuffle.input[0];chain=[]
    while True:
        n=producers.get(value)
        if n is None:raise ValueError('Missing output projection')
        if n.op_type=='Conv':break
        if n.op_type not in ('Cast','Identity','Clip','Mul'):raise ValueError('Unsupported normalized output path')
        dynamic=[i for i in n.input if i and i not in constants]
        if len(dynamic)!=1 or any(constants[i].size!=1 for i in n.input if i in constants):raise ValueError('Only scalar pointwise path supported')
        chain.append((n,dynamic[0]));value=dynamic[0]
    conv=n;a=attributes(conv)
    if conv.input[1] not in constants:raise ValueError('Unknown projection weights')
    weight=constants[conv.input[1]]
    if weight.shape!=(36,16,3,3) or a.get('group',1)!=1 or a.get('pads',[0]*4)!=[1]*4 or a.get('strides',[1,1])!=[1,1] or a.get('dilations',[1,1])!=[1,1] or a.get('auto_pad',b'NOTSET')!=b'NOTSET':raise ValueError('Require unchanged Conv16-to-36 k3 stride1 pad1')
    if len(conv.input)!=3 or conv.input[2] not in constants:raise ValueError('Require explicit known 36-phase bias')
    bias=constants[conv.input[2]]
    if bias.shape!=(36,) or weight.dtype not in (np.float16,np.float32) or bias.dtype!=weight.dtype:raise ValueError('Invalid projection type/bias')
    inferred=onnx.shape_inference.infer_shapes(model);info={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)}
    v=info.get(conv.input[0]);shape=[] if v is None else [d.dim_value for d in v.type.tensor_type.shape.dim]
    if len(shape)!=4 or shape[1]!=16 or shape[2]<=0 or shape[3]<=0:raise ValueError('Need static spatial feature shape')
    if shape[0]==0:
        if any(v.type.tensor_type.shape.dim[0].dim_value!=1 for v in model.graph.input):raise ValueError('Unknown batch without static inputs')
        shape[0]=1
    if shape[0]!=1:raise ValueError('Require batch one')
    removed=[conv,shuffle]+[n for n,_ in chain];dead_values={o for n in removed for o in n.output}
    for n in model.graph.node:
        if n not in removed and any(i in dead_values and i!=shuffle.output[0] for i in n.input):raise ValueError('Projection path has another consumer')
    rest=6//factor
    reduced=np.zeros((factor*factor,16,3,3),dtype=weight.dtype);reduced_bias=np.zeros(factor*factor,dtype=bias.dtype)
    grid=weight.reshape(6,6,16,3,3).astype(np.float32);bias_grid=bias.reshape(6,6).astype(np.float32)
    for y in range(factor):
        for x in range(factor):
            reduced[y*factor+x]=grid[y*rest:(y+1)*rest,x*rest:(x+1)*rest].mean((0,1)).astype(weight.dtype)
            reduced_bias[y*factor+x]=bias_grid[y*rest:(y+1)*rest,x*rest:(x+1)*rest].mean().astype(bias.dtype)
    if mode in ('preserve_nearest','preserve_fixed'):
        if factor!=3:raise ValueError('Native-mean constraint requires nine phases')
        import sys
        sys.path.insert(0,str(Path(__file__).parent.parent/'runtime'))
        from phase_projection import preserve_native_means
        reduced,reduced_bias,_=preserve_native_means(weight,bias)
    if projection_source is not None:
        if factor!=3 or mode not in ('preserve_nearest','preserve_fixed') or frozen_source is None:
            raise ValueError('Trained projection requires constrained nine phases and frozen source')
        def learned_arrays(graph,channels):
            graph,_=materialize(graph)
            ci={i.name:numpy_helper.to_array(i) for i in graph.graph.initializer}
            candidates=[n for n in graph.graph.node if n.op_type=='Conv' and len(n.input)==3 and n.input[1] in ci and ci[n.input[1]].shape==(channels,16,3,3)]
            if len(candidates)!=1:raise ValueError('Require unique learned output projection')
            node=candidates[0];attrs=attributes(node)
            if attrs.get('group',1)!=1 or attrs.get('pads',[0]*4)!=[1]*4 or attrs.get('strides',[1,1])!=[1,1] or attrs.get('dilations',[1,1])!=[1,1]:
                raise ValueError('Unsupported trained projection attributes')
            return ci[node.input[1]],ci[node.input[2]]
        fw,fb=learned_arrays(frozen_source,36)
        if not np.array_equal(fw.astype(weight.dtype),weight) or not np.array_equal(fb.astype(bias.dtype),bias):
            raise ValueError('Native output weights differ from frozen source')
        pw,pb=learned_arrays(projection_source,9)
        if pb.shape!=(9,) or not np.isfinite(pw).all() or not np.isfinite(pb).all():
            raise ValueError('Invalid trained output projection')
        reduced,reduced_bias=pw.astype(weight.dtype),pb.astype(bias.dtype)
    interpolation='nearest' if mode=='preserve_nearest' else mode
    def init(label,x):
        name=prefix+label;model.graph.initializer.append(numpy_helper.from_array(x,name));return name
    extra=[helper.make_node('Conv',[conv.input[0],init('weight',reduced),init('bias',reduced_bias)],[prefix+'projected'],kernel_shape=[3,3],pads=[1]*4)]
    current=prefix+'projected'
    if mode in ('fixed','preserve_fixed'):
        kernel=np.zeros((factor*factor,1,6,6),dtype=weight.dtype)
        for y in range(factor):
            for x in range(factor):kernel[y*factor+x,0,y*rest:(y+1)*rest,x*rest:(x+1)*rest]=1
        extra.append(helper.make_node('ConvTranspose',[current,init('fixed_kernel',kernel)],[prefix+'expanded'],kernel_shape=[6,6],strides=[6,6]));current=prefix+'expanded'
    else:
        if factor!=1:
            extra.append(helper.make_node('DepthToSpace',[current],[prefix+'spatial'],blocksize=factor,mode='CRD'));current=prefix+'spatial'
        extra.append(helper.make_node('Resize',[current,'',init('scales',np.array([1,1,rest,rest],np.float32))],[prefix+'expanded'],mode='nearest' if interpolation=='nearest' else 'linear',coordinate_transformation_mode='asymmetric' if interpolation=='nearest' else 'half_pixel',nearest_mode='floor'))
        current=prefix+'expanded'
    for j,(n,operand) in enumerate(reversed(chain)):
        new=copy.deepcopy(n);new.name=prefix+f'pointwise_{j}'
        for i,v in enumerate(new.input):
            if v==operand:new.input[i]=current
            elif v in constants:new.input[i]=init(f'constant_{j}_{i}',constants[v])
        current=prefix+f'pointwise_value_{j}';new.output[0]=current;extra.append(new)
    extra.append(helper.make_node('Identity',[current],[shuffle.output[0]]))
    nodes=[]
    for n in model.graph.node:
        if n==shuffle:nodes.extend(extra)
        if n not in removed:nodes.append(n)
    del model.graph.node[:];model.graph.node.extend(nodes)
    lookup={o:n for n in model.graph.node for o in n.output};live=set()
    def visit(value):
        if value in live:return
        live.add(value)
        if value in lookup:
            for i in lookup[value].input:
                if i:visit(i)
    for v in model.graph.output:visit(v.name)
    nodes=[n for n in model.graph.node if any(o in live for o in n.output)];inits=[v for v in model.graph.initializer if v.name in live];del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.initializer[:];model.graph.initializer.extend(inits);del model.graph.value_info[:];onnx.checker.check_model(model);return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--factor',type=int,required=True,choices=[1,2,3]);p.add_argument('--mode',choices=['nearest','bilinear','fixed','preserve_nearest','preserve_fixed'],default='nearest');a=p.parse_args();path=Path(a.output)
    if path.exists() or path.with_suffix('.phases.json').exists():raise FileExistsError(path)
    model=reduce_phases(onnx.load(a.input),a.factor,a.mode);path.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,path);path.with_suffix('.phases.json').write_text(json.dumps({'source':a.input,'factor':a.factor,'interpolation':a.mode,'independent_phases':a.factor*a.factor,'approximate':True,'full_output_retained':True,'SDK_compiled':False,'NPU_measured':False},indent=2))
if __name__=='__main__':main()
