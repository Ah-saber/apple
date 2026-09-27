"""Build searched ONNX graphs from verified exports and restricted checkpoint reader.
No Torch dependency: the only accepted pickle globals are HalfStorage,
_rebuild_tensor_v2, and OrderedDict. No arbitrary pickle classes execute.
"""
import argparse,collections,io,json,pickle,sys,zipfile
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
base=Path(__file__).parent;sys.path.insert(0,str(base/'tools'))
from reduce_output_phases import reduce_phases
from rewrite_hotspots import rewrite_statistics
from rewrite_reference_relu import rewrite as rewrite_ref

def checkpoint_arrays(path):
 with zipfile.ZipFile(path) as archive:
  name=next(n for n in archive.namelist() if n.endswith('/data.pkl'));root=name.rsplit('/',1)[0]
  def rebuild(storage,offset,size,stride,requires_grad,hooks):
   arr=np.asarray(storage,dtype=np.float16)
   if any(v<0 for v in stride) or offset<0:raise ValueError('Invalid strides')
   final=offset+sum((s-1)*t for s,t in zip(size,stride))
   if final>=arr.size:raise ValueError('Tensor exceeds storage')
   return np.lib.stride_tricks.as_strided(arr[offset:],shape=size,strides=tuple(t*2 for t in stride)).copy()
  class Reader(pickle.Unpickler):
   def find_class(self,module,name):
    if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
    if (module,name)==('torch','HalfStorage'):return 'half'
    if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
    raise ValueError('Unsupported pickle global: '+module+'.'+name)
   def persistent_load(self,pid):
    if pid[0]!='storage' or pid[1]!='half' or not str(pid[2]).isdigit():raise ValueError('Unsupported storage')
    arr=np.frombuffer(archive.read(root+'/data/'+str(pid[2])),dtype='<f2').copy()
    if arr.size!=pid[4]:raise ValueError('Storage size mismatch')
    return arr
  return Reader(io.BytesIO(archive.read(name))).load()['projection']

def set_projection(g,arrays):
 lookup={v.name:v for v in g.graph.initializer};matches=[n for n in g.graph.node if n.op_type=='Conv' and n.input[1] in lookup and numpy_helper.to_array(lookup[n.input[1]]).shape==(9,16,3,3)]
 if len(matches)!=1:raise ValueError('Need unique nine-phase projection')
 n=matches[0]
 for index,key in [(1,'weight'),(2,'bias')]:
  value=arrays[key];old=numpy_helper.to_array(lookup[n.input[index]])
  if value.shape!=old.shape or value.dtype!=old.dtype:raise ValueError('Trained array shape/type mismatch')
  lookup[n.input[index]].CopyFrom(numpy_helper.from_array(value,n.input[index]))
 return g

def main():
 rows=[]
 for scene in ('ordinary','special'):
  arrays=checkpoint_arrays(base/'models'/f'{scene}_output_quantsearch.pt')
  for small in (True,False):
   suffix='_small' if small else '';folder=base/'source_onnx';old=onnx.load(folder/f'{scene}_baseline{suffix}.onnx')
   source=set_projection(onnx.load(folder/f'{scene}_phase3_preserve_nearest{suffix}.onnx'),arrays)
   cases=['phase3_quantsearch_nearest','phase3_quantsearch_fixed','space_pack_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_nearest','space_pack_reference_relu_output_stable_phase3_quantsearch_fixed']
   for case in cases:
    # Original exported Torch graph with only the validated projection arrays changed.
    original_case=case.replace('quantsearch','preserve')
    if original_case.endswith('fixed') and original_case.startswith('space_pack_reference'):
     # Source reference graph has identical feature path; output expansion is built separately.
     source_case='space_pack_reference_relu_output_stable_phase3_preserve_nearest'
     g=onnx.load(folder/f'{scene}_{source_case}{suffix}.onnx')
     # Both equivalent output representations are verified against the nearest source.
     src=set_projection(g,arrays)
    else:src=set_projection(onnx.load(folder/f'{scene}_{original_case}{suffix}.onnx'),arrays)
    target=folder/f'{scene}_{case}{suffix}.onnx'
    if target.exists():raise FileExistsError(target)
    onnx.checker.check_model(src);onnx.save(src,target)
    rewritten=old
    if case.startswith('space_pack'):rewritten=rewrite_statistics(rewritten,'space_pack')
    if 'reference_relu' in case:rewritten,_=rewrite_ref(rewritten,old,onnx.load(folder/f'{scene}_reference_relu_output_stable{suffix}.onnx'))
    rewritten=reduce_phases(rewritten,3,'preserve_fixed' if case.endswith('fixed') else 'preserve_nearest',source,old)
    path=folder/f'{scene}_{case}_rewritten{suffix}.onnx'
    if path.exists():raise FileExistsError(path)
    onnx.save(rewritten,path);rows.append({'scene':scene,'case':case,'small':small,'source':target.name,'rewritten':path.name,'SDK_verified':False,'source_equivalence_note':'Combined fixed graph uses nearest source control, identical phase-to-pixel mapping'})
 (base/'evidence/searched_local_export_manifest.json').write_text(json.dumps({'models':rows,'restricted_checkpoint_globals':True,'NPU_verified':False},indent=2))
 print('BUILT',len(rows)*2,'graphs',flush=True)
if __name__=='__main__':main()
