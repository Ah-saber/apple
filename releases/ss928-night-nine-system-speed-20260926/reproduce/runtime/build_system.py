import json
from pathlib import Path
import torch
from load_packed_model import load_packed_model
from system_variants import QuarterFront,CompactReference,CompactBody,FactorizedOutput,FoldedFront,FullSystem,LowScaleSystem,PhaseRepeatSystem,DirectHeadSystem,LinearFront,RowGroupedSystem,StatsFront,CenteredStatsFront,SiLUReference,BalancedCenteredFront,ContrastStatsFront

def load_system(root,scene,variant):
 root=Path(root);run=root/'runs'/('SS928-PACKED-FRONT-COMPACT8-20260926' if scene=='ordinary' else 'SS928-PACKED-FRONT-20260926');step=json.loads((run/f'{scene}_selection.json').read_text())['selected_step'];base=load_packed_model(run/f'{scene}_packed_front_{step:06d}.pt',root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{scene}_factor24.pt',root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{scene}_fused.pt');out=root/'runs/SS928-SYSTEM-SPEED-20260926';front=None;reference=None
 selection=json.loads((out/f'{scene}_selection.json').read_text())
 if variant.startswith('linear'):
  kernel=4 if 'k4' in variant else 2;ck=torch.load(out/f'{scene}_linear_k{kernel}.pt',map_location='cpu',weights_only=True);front=LinearFront(kernel);front.load_state_dict(ck['front']);front=front.cuda().to(dtype=torch.float32 if 'f32' in variant else torch.float16,memory_format=torch.channels_last)
 if variant.startswith('balanced'):
  width=8 if scene=='ordinary' else 12;ck=torch.load(out/f'{scene}_balanced_w{width}_003000.pt',map_location='cpu',weights_only=True);
  if 'tuned' in variant:ck=torch.load(out/f'{scene}_front_output_003000.pt',map_location='cpu',weights_only=True)
  front=BalancedCenteredFront(width);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('contrast'):
  width=8 if scene=='ordinary' else 12;ck=torch.load(out/f'{scene}_contrast_w{width}_003000.pt',map_location='cpu',weights_only=True);front=ContrastStatsFront(width);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('centered'):
  width=16 if 'w16' in variant else (8 if scene=='ordinary' else 12);ck=torch.load(out/f'{scene}_centered_w{width}_003000.pt',map_location='cpu',weights_only=True);front=CenteredStatsFront(width);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last)
 if 'silu' in variant or 'sigref' in variant:
  reference=SiLUReference(base.core.model.global_reference).cuda().half().to(memory_format=torch.channels_last)
  if 'sigref' in variant:
   ck=torch.load(out/f'{scene}_silu_002000.pt',map_location='cpu',weights_only=True);reference.load_state_dict(ck['reference'])
 if 'refit' in variant:
  ck=torch.load(out/f'{scene}_reference_postfit_0.01.pt',map_location='cpu',weights_only=True);reference=CompactReference(ck['width']);reference.load_state_dict(ck['reference']);reference=reference.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('stats'):
  width=16 if 'w16' in variant else (8 if scene=='ordinary' else 12);ck=torch.load(out/f'{scene}_stats_w{width}_003000.pt',map_location='cpu',weights_only=True);front=StatsFront(width);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('all') or variant.startswith('quarter') or variant.startswith('direct_all'):
  ck=torch.load(out/selection['selected_front'],map_location='cpu',weights_only=True);front=QuarterFront(ck['width'],ck['blocks']);front.load_state_dict(ck['front']);front=front.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('all') or variant.startswith('reference') or variant.startswith('quarter_ref') or variant.startswith('direct_all'):
  ck=torch.load(out/selection['selected_reference'],map_location='cpu',weights_only=True);reference=CompactReference(ck['width']);reference.load_state_dict(ck['reference']);reference=reference.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('body') or '_body' in variant:
  ck=torch.load(out/f'{scene}_body_output_002000.pt' if 'refined' in variant else out/selection['selected_body'],map_location='cpu',weights_only=True);body=CompactBody(ck['blocks']);body.load_state_dict(ck['body']);base.core.model.body=body.cuda().half().to(memory_format=torch.channels_last)
 if variant.startswith('factor') or 'rank' in variant:
  rank=8 if variant.endswith('8') else 12;base.core.model.upsample[0]=FactorizedOutput(base.core.model.upsample[0],rank)
 if 'weighted' in variant:
  rank=8 if 'weighted8' in variant else 12;ck=torch.load(out/f'{scene}_weighted_r{rank}_003000.pt',map_location='cpu',weights_only=True);layer=FactorizedOutput(base.core.model.upsample[0],rank);layer.load_state_dict(ck['output']);base.core.model.upsample[0]=layer
 if variant.startswith('fold'):front=FoldedFront(base.front,'float32' if variant=='fold32' else 'float16')
 dtype='float16' if variant.endswith('_half') or variant=='all_half_io' else ('uint8' if variant.endswith('_byte') or variant=='pre_byte' else 'float32')
 if 'rows' in variant or variant.startswith('group'):
  groups=int(variant.rsplit('rows',1)[1]) if 'rows' in variant else int(variant.split('group',1)[1]);m=RowGroupedSystem(base,front,reference,groups,dtype,'low' in variant)
 elif 'phase' in variant:m=PhaseRepeatSystem(base,front,reference,dtype)
 elif variant.startswith('direct'):m=DirectHeadSystem(base,front,reference,True,dtype)
 elif 'low' in variant or variant in ['pre_byte','all_half_io','all_byte','all_pre_byte']:m=LowScaleSystem(base,front,reference,dtype,variant in ['pre_byte','all_pre_byte'])
 else:m=FullSystem(base,front,reference,output_mode=variant if variant.startswith('transpose') else 'shuffle',output_dtype=dtype)
 return m.eval()

def load_case(root,scene,case):
 from system_variants import InputLayoutSystem
 core=case;dtype=None
 for suffix,value in [('_half','float16'),('_byte','uint8')]:
  if core.endswith(suffix):core=core[:-len(suffix)];dtype=value;break
 layout='NHWC' if '_nhwc16' in core else 'NCHW';half='_nhwc16' in core or '_nchw16' in core;core=core.replace('_nhwc16','').replace('_nchw16','');m=load_system(root,scene,core)
 if dtype is not None:m.output_dtype=dtype
 if layout=='NHWC':m=InputLayoutSystem(m)
 return m.eval(),{'raw_input_layout':layout,'raw_input_dtype':'float16' if half else 'float32','context_input_dtype':'float32','output_dtype':m.output_dtype,'core_variant':core}

def case_inputs(x,ctx,info):
 x=x.half() if info['raw_input_dtype']=='float16' else x
 if info['raw_input_layout']=='NHWC':x=x.permute(0,2,3,1).contiguous()
 return x,ctx
