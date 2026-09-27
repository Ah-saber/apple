"""Bounded ONNX rewrites of v0.7 fixed statistics and complete shuffle6.
Keeps the deployment-side native R2 reference/body compatibility paths.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

def attrs(node):
    return {a.name:helper.get_attribute_value(a) for a in node.attribute}

def constants(model):
    result={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type=='Constant' and 'value' in attrs(node):
            result[node.output[0]]=numpy_helper.to_array(attrs(node)['value'])
    return result

def start(model,prefix):
    model=copy.deepcopy(model)
    if any(v.startswith(prefix) for n in model.graph.node for v in list(n.input)+list(n.output)):
        raise ValueError('Rewrite namespace already exists')
    return model

def replace(model,old,nodes):
    result=[]
    for n in model.graph.node:
        result.extend(nodes if n==old else [n])
    del model.graph.node[:];model.graph.node.extend(result)
    # Remove metadata so old inferred intermediate shapes cannot constrain new nodes.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model

def rewrite_statistics(model,mode):
    prefix='v07_stats__';model=start(model,prefix);const=constants(model)
    found=[]
    for n in model.graph.node:
        if n.op_type=='Conv' and n.input[1] in const:
            w=const[n.input[1]];a=attrs(n)
            if w.shape==(12,9,2,2) and a.get('strides',[1,1])==[2,2] and not any(a.get('pads',[0,0,0,0])) and a.get('group',1)==1:
                found.append(n)
    if len(found)!=1:raise ValueError('Require unique fixed 9-to-12 k2 stride2 statistics convolution')
    old=found[0];w=const[old.input[1]]
    if len(old.input)>2 and old.input[2] and (old.input[2] not in const or np.any(const[old.input[2]]!=0)):
        raise ValueError('Nonzero/unknown statistics bias')
    temporal=np.stack([w[4*i,:,0,0] for i in range(3)])[:,:,None,None]
    expected=np.zeros_like(w)
    for g in range(3):
        for dy in range(2):
            for dx in range(2):expected[4*g+2*dy+dx,:,dy,dx]=temporal[g,:,0,0]
    if not np.array_equal(expected,w):raise ValueError('Kernel is not pure packed statistics')
    expected_temporal=np.zeros((3,9),dtype=w.dtype)
    expected_temporal[0,8]=1;expected_temporal[1,:5]=120/727;expected_temporal[1,5]=127/727
    expected_temporal[2,6:8]=123/373;expected_temporal[2,8]=127/373
    if not (np.array_equal(temporal[:,:,0,0],expected_temporal) or np.array_equal(temporal[:,:,0,0],expected_temporal.astype(np.float16).astype(w.dtype))):
        raise ValueError('Unrecognized balanced coefficients; reject learned or contrast kernel')
    def init(label,value):
        name=prefix+label;model.graph.initializer.append(numpy_helper.from_array(value,name));return name
    nodes=[];source=old.input[0];dest=old.output[0]
    if mode=='temporal_f32':
        source=prefix+'input_float';nodes.append(helper.make_node('Cast',[old.input[0]],[source],to=TensorProto.FLOAT))
        temporal=temporal.astype(np.float32);dest=prefix+'packed_float'
    if mode in ('temporal_first','temporal_f32'):
        pack=np.zeros((12,1,2,2),dtype=temporal.dtype)
        for g in range(3):
            for dy in range(2):
                for dx in range(2):pack[4*g+2*dy+dx,0,dy,dx]=1
        nodes.extend([helper.make_node('Conv',[source,init('temporal',temporal)],[prefix+'temporal_out'],kernel_shape=[1,1]),helper.make_node('Conv',[prefix+'temporal_out',init('pack',pack)],[dest],kernel_shape=[2,2],strides=[2,2],group=3)])
        if mode=='temporal_f32':nodes.append(helper.make_node('Cast',[dest],[old.output[0]],to=TensorProto.FLOAT16 if w.dtype==np.float16 else TensorProto.FLOAT))
    elif mode in ('space_pack','temporal_space_pack'):
        if mode=='space_pack':
            mix=np.zeros((12,36,1,1),dtype=w.dtype)
            for g in range(3):
                for p in range(4):mix[4*g+p,p*9:p*9+9,0,0]=temporal[g,:,0,0]
            nodes.extend([helper.make_node('SpaceToDepth',[source],[prefix+'packed9'],blocksize=2),helper.make_node('Conv',[prefix+'packed9',init('mix',mix)],[dest],kernel_shape=[1,1])])
        else:
            order=np.array([p*3+g for g in range(3) for p in range(4)],np.int64)
            nodes.extend([helper.make_node('Conv',[source,init('temporal',temporal)],[prefix+'temporal_out'],kernel_shape=[1,1]),helper.make_node('SpaceToDepth',[prefix+'temporal_out'],[prefix+'packed3'],blocksize=2),helper.make_node('Gather',[prefix+'packed3',init('order',order)],[dest],axis=1)])
    elif mode=='pack_first':
        pack=np.zeros((36,1,2,2),dtype=w.dtype);mix=np.zeros((12,36,1,1),dtype=w.dtype)
        for dy in range(2):
            for dx in range(2):
                p=2*dy+dx
                for t in range(9):pack[4*t+p,0,dy,dx]=1
                for g in range(3):mix[4*g+p,p::4,0,0]=temporal[g,:,0,0]
        nodes.extend([helper.make_node('Conv',[source,init('pack9',pack)],[prefix+'packed9'],kernel_shape=[2,2],strides=[2,2],group=9),helper.make_node('Conv',[prefix+'packed9',init('mix',mix)],[dest],kernel_shape=[1,1])])
    else:raise ValueError(mode)
    return replace(model,old,nodes)

def rewrite_output(model,mode):
    prefix='v07_output__';model=start(model,prefix)
    matches=[n for n in model.graph.node if n.op_type=='DepthToSpace' and attrs(n).get('blocksize')==6]
    if len(matches)!=1:raise ValueError('Require unique native shuffle6')
    old=matches[0]
    if attrs(old).get('mode',b'DCR')!=b'CRD':raise ValueError('Require CRD output channel order')
    inferred=onnx.shape_inference.infer_shapes(model)
    info={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)+list(inferred.graph.output)}
    v=info.get(old.input[0])
    if v is None:raise ValueError('No inferred packed shape/type')
    shape=[d.dim_value for d in v.type.tensor_type.shape.dim];typ=v.type.tensor_type.elem_type
    if len(shape)==4 and shape[0]==0:
        output_shapes=[[d.dim_value for d in out.type.tensor_type.shape.dim] for out in model.graph.output]
        if output_shapes!=[[1,1,shape[2]*6,shape[3]*6]] or any(inp.type.tensor_type.shape.dim[0].dim_value!=1 for inp in model.graph.input):
            raise ValueError('Unknown batch without static full-output/input proof')
        shape[0]=1
    if len(shape)!=4 or shape[:2]!=[1,36] or not all(shape):raise ValueError('Require static batch-one 36-phase output')
    if typ not in (TensorProto.FLOAT,TensorProto.FLOAT16):raise ValueError('Require floating packed output')
    h,w=shape[-2:];dtype=np.float16 if typ==TensorProto.FLOAT16 else np.float32
    def init(label,value):
        name=prefix+label;model.graph.initializer.append(numpy_helper.from_array(value,name));return name
    source,target=old.input[0],old.output[0];nodes=[]
    if mode=='reshape':
        nodes=[helper.make_node('Reshape',[source,init('packed_shape',np.array([6,6,h,w],np.int64))],[prefix+'phases']),helper.make_node('Transpose',[prefix+'phases'],[prefix+'spatial'],perm=[2,0,3,1]),helper.make_node('Reshape',[prefix+'spatial',init('full_shape',np.array([1,1,h*6,w*6],np.int64))],[target])]
    elif mode in ('deconv6','deconv2_shuffle3','deconv3_shuffle2'):
        factor={'deconv6':6,'deconv2_shuffle3':2,'deconv3_shuffle2':3}[mode];rest=6//factor
        weight=np.zeros((36,rest*rest,factor,factor),dtype=dtype)
        for dy in range(factor):
            for dx in range(factor):
                for sy in range(rest):
                    for sx in range(rest):weight[(rest*dy+sy)*6+rest*dx+sx,rest*sy+sx,dy,dx]=1
        dest=target if rest==1 else prefix+'expanded'
        nodes=[helper.make_node('ConvTranspose',[source,init('fixed_kernel',weight)],[dest],kernel_shape=[factor,factor],strides=[factor,factor])]
        if rest!=1:nodes.append(helper.make_node('DepthToSpace',[dest],[target],blocksize=rest,mode='CRD'))
    elif mode in ('axis_horizontal','axis_vertical'):
        horizontal=mode=='axis_horizontal'
        first=np.zeros((36,6,1,6) if horizontal else (36,6,6,1),dtype=dtype)
        second=np.zeros((6,1,6,1) if horizontal else (6,1,1,6),dtype=dtype)
        for dy in range(6):
            for dx in range(6):
                if horizontal:first[dy*6+dx,dy,0,dx]=1
                else:first[dy*6+dx,dx,dy,0]=1
        for p in range(6):
            if horizontal:second[p,0,p,0]=1
            else:second[p,0,0,p]=1
        k1,k2=([1,6],[6,1]) if horizontal else ([6,1],[1,6])
        nodes=[helper.make_node('ConvTranspose',[source,init('axis_first',first)],[prefix+'axis_middle'],kernel_shape=k1,strides=k1),helper.make_node('ConvTranspose',[prefix+'axis_middle',init('axis_second',second)],[target],kernel_shape=k2,strides=k2)]
    elif mode in ('shuffle2_3','shuffle3_2'):
        factor=2 if mode=='shuffle2_3' else 3;rest=6//factor
        ids=np.array([(rest*dy+sy)*6+rest*dx+sx for sy in range(rest) for sx in range(rest) for dy in range(factor) for dx in range(factor)],np.int64)
        nodes=[helper.make_node('Gather',[source,init('channel_order',ids)],[prefix+'ordered'],axis=1),helper.make_node('DepthToSpace',[prefix+'ordered'],[prefix+'stage1'],blocksize=factor,mode='CRD'),helper.make_node('DepthToSpace',[prefix+'stage1'],[target],blocksize=rest,mode='CRD')]
    else:raise ValueError(mode)
    return replace(model,old,nodes)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--statistics',choices=['temporal_first','temporal_f32','pack_first','space_pack','temporal_space_pack']);p.add_argument('--permutation',choices=['reshape','deconv6','deconv2_shuffle3','deconv3_shuffle2','shuffle2_3','shuffle3_2','axis_horizontal','axis_vertical']);a=p.parse_args()
    if not(a.statistics or a.permutation):p.error('Choose at least one bounded rewrite')
    target=Path(a.output)
    if target.exists() or target.with_suffix('.rewrite.json').exists():raise FileExistsError(target)
    model=onnx.load(a.input)
    if a.statistics:model=rewrite_statistics(model,a.statistics)
    if a.permutation:model=rewrite_output(model,a.permutation)
    onnx.checker.check_model(model);target.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,target)
    target.with_suffix('.rewrite.json').write_text(json.dumps({'source':str(a.input),'statistics':a.statistics,'permutation':a.permutation,'full_model_retained':True,'SDK_compiled':False,'NPU_measured':False},indent=2))
if __name__=='__main__':main()
