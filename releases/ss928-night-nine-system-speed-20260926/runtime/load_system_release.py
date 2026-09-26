"""Load frozen v0.6 dependency plus explicitly selected new trained front."""
import hashlib,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from system_variants import FullSystem,RowGroupedSystem,InputLayoutSystem,BalancedCenteredFront,ContrastStatsFront,SiLUReference

def load_release(scene,case='primary',device='cuda',release_dir=None,base_dir=None):
 if scene not in ('ordinary','special'):raise ValueError('Invalid scene')
 release=Path(release_dir or Path(__file__).resolve().parents[1]);base=Path(base_dir or release.parent/'ss928-night-nine-packed-front-20260926');sys.path.insert(0,str(base/'runtime'))
 from load_packed_model import load_packed_model
 frontpath=base/'models'/f'{scene}_packed_joint{8 if scene=="ordinary" else 12}.pt';frozen=load_packed_model(frontpath,base/'models'/f'{scene}_factor24.pt',base/'models'/f'{scene}_fused.pt',device=device)
 mapping={'base':'base','rows_only':'base_rows16','front_f32':'balanced','front_rows_f32':'balanced_rows16','primary':'balanced_low_rows16_nhwc16_half','byte':'balanced_low_rows16_nhwc16_byte','contrast_f32':'contrast','contrast_rows_f32':'contrast_rows16','contrast_half':'contrast_low_rows16_nhwc16_half','reference_only':'sigref','primary_reference':'balanced_sigref_low_rows16_nhwc16_half'}
 if case not in mapping:raise ValueError(f'Unknown case: {case}')
 variant=mapping[case];front=None;reference=None
 if variant.startswith(('balanced','contrast')):
  kind='contrast' if variant.startswith('contrast') else 'balanced';width=8 if scene=='ordinary' else 12;ck=torch.load(release/'models'/f'{scene}_{kind}_w{width}_003000.pt',map_location='cpu',weights_only=True)
  if hashlib.sha256(frontpath.read_bytes()).hexdigest()!=ck['teacher_sha256']:raise ValueError('Frozen packed-front teacher hash mismatch')
  front=(ContrastStatsFront if kind=='contrast' else BalancedCenteredFront)(width);front.load_state_dict(ck['front'],strict=True);front=front.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
 if 'sigref' in variant:
  reference=SiLUReference(frozen.core.model.global_reference);ck=torch.load(release/'models'/f'{scene}_silu_002000.pt',map_location='cpu',weights_only=True);reference.load_state_dict(ck['reference'],strict=True);reference=reference.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
 half=variant.endswith('_half');dtype='uint8' if variant.endswith('_byte') else ('float16' if half else 'float32')
 if 'rows16' in variant:model=RowGroupedSystem(frozen,front,reference,16,dtype,'low' in variant)
 else:model=FullSystem(frozen,front,reference,output_dtype=dtype)
 nhwc='_nhwc16' in variant
 if nhwc:model=InputLayoutSystem(model)
 return model.eval(),{'scene':scene,'case':case,'variant':variant,'raw_layout':'NHWC' if nhwc else 'NCHW','raw_dtype':'float16' if nhwc else 'float32','context_dtype':'float32','output_dtype':dtype,'output_shape':[1,1,3072,3840]}

def prepare_inputs(stack,context,info,device):
 if tuple(stack.shape)!=(1,9,1024,1280) or tuple(context.shape)!=(1,1,64,64):raise ValueError('Expected complete preprocessed nine-frame vectors')
 stack=stack.to(device=device,dtype=torch.float16 if info['raw_dtype']=='float16' else torch.float32);context=context.to(device=device,dtype=torch.float32)
 if info['raw_layout']=='NHWC':stack=stack.permute(0,2,3,1).contiguous()
 return stack,context
