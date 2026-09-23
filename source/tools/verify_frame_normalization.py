"""Regression checks against baseline geometry, physical units and frozen teacher."""
import sys,json,time,argparse,copy
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ir_sr.training import dataset_for_config,learning_rate,atomic_json
from ir_sr.sequence_normalization import calibrate_sequence,SequenceNormalization,digest
from ir_sr.auxiliary import loss_terms
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());old=copy.deepcopy(c)
for k in ['sequence_normalization_index','sequence_normalization_sha256']:old.pop(k)
old['max_steps']=200000;old.pop('lr_schedule_steps',None);torch.set_num_threads(2);start=time.perf_counter();checks=[]
for step in [1,200,500,2000,8000]:assert learning_rate(step,c)==learning_rate(step,old)
for split in ['train','val','test']:
 before=dataset_for_config(old,split);after=dataset_for_config(c,split);groups={}
 for i,r in enumerate(after.records):groups.setdefault(r['sequence_id'],[]).append(i)
 for ids in groups.values():
  for i in [ids[0],ids[-1]]:
   x,y=before[i],after[i];r=after.records[i];n=after.normalization_for(r)
   assert x['sample_id']==y['sample_id'] and x['crop_tlhw']==y['crop_tlhw'] and x['augmentation_id']==y['augmentation_id']
   assert torch.equal(x['gt'],y['gt'])
   dn0=x['raw']*109.56366230459803+5079.22721852674;dn1=y['raw']*n['scale']+n['offset'];err=float((dn0-dn1).abs().max());assert err<.002,err
   if split=='train':
    dn0=x['middle_raw']*109.56366230459803+5079.22721852674;dn1=y['middle_raw']*n['scale']+n['middle_offset'];assert float((dn0-dn1).abs().max())<.002
    assert np.isclose(y['raw_loss_multiplier'],n['scale']/109.56366230459803)
   else:
    raw=after._raw(r).astype(np.float32);assert np.array_equal(after.full_raw(i).numpy()[0],(raw-n['offset'])/n['scale'])
   checks.append(r['sample_id'])
# Known physical DN error must have unchanged auxiliary loss across scales.
class Dummy(torch.nn.Module):
 def __init__(self):super().__init__();self.dn_error=torch.nn.Parameter(torch.tensor(1.))
 def forward(self,x,return_auxiliary=False):return x*0,x+self.dn_error/scales[:,None,None,None]
scales=torch.tensor([60.,200.]);model=Dummy();x=torch.zeros(2,1,12,12);total,display,aux=loss_terms(model,x,x,x,.1,scales/109.56366230459803)
assert abs(float(aux)-1/109.56366230459803)<1e-8;total.backward();assert torch.isfinite(model.dn_error.grad)
# Sequence-wide affine transform should not alter normalized values (constant scale fixture).
raw=(5000+np.arange(80*96).reshape(80,96)%100).astype(np.float32);stack=np.stack([raw,raw+2,raw-1]);params,segs=calibrate_sequence(stack);shifted,_=calibrate_sequence(stack+10)
for i,(u,v) in enumerate(zip(params,shifted)):assert np.allclose((stack[i]-u['offset'])/u['scale'],(stack[i]+10-v['offset'])/v['scale'],atol=1e-5)
# Metadata checks must reject calibration misuse.
n=SequenceNormalization(c['sequence_normalization_index'],c['sequence_normalization_sha256']);r=copy.deepcopy(after.records[0]);r['split']='train'
try:n.for_record(r);raise AssertionError('split mismatch accepted')
except ValueError:pass
try:SequenceNormalization(c['sequence_normalization_index'],'0'*64);raise AssertionError('hash mismatch accepted')
except ValueError:pass
report={'status':'passed','paired_samples':len(checks),'checks':['identical crop/GT/augmentation','invertible input and auxiliary physical DN coordinates','auxiliary1DN loss scale invariant','8000-step LR matches200000-step baseline prefix','validation/native input calibration','constant offset invariance','split/hash misuse rejected'],'samples':checks,'seconds':time.perf_counter()-start};atomic_json(a.output,report);print(json.dumps(report))
