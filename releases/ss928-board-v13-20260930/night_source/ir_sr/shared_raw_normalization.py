"""Explicit same-video, full-RAW calibration shared across label partitions.

This is a separate engineering protocol. It does not satisfy RAW-disjoint
spatial validation, and must not silently replace SequenceNormalization.
"""
from pathlib import Path
import json,numpy as np
from ir_sr.sequence_normalization import digest
class SharedRawNormalization:
 def __init__(self,path,expected_sha256):
  if not expected_sha256 or digest(path)!=expected_sha256:raise ValueError('Shared RAW calibration identity mismatch')
  self.index=json.loads(Path(path).read_text());self.sha256=expected_sha256
  if self.index.get('status')!='complete' or self.index.get('kind')!='shared_full_raw_sequence_v1':raise ValueError('Explicit shared full RAW protocol required')
  self.by_key={r['key']:r for r in self.index['sequences']}
  self.fields={};self.dynamic={}
  for key,r in self.by_key.items():
   if 'fixed_field' in r:
    f=r['fixed_field'];fp=Path(path).parent/f['path']
    if digest(fp)!=f['sha256']:raise ValueError('Fixed field identity mismatch')
    field=np.load(fp,allow_pickle=False)
    if field.dtype!=np.float32 or list(field.shape)!=r['shape'][1:] or not np.isfinite(field).all():raise ValueError('Invalid fixed field')
    field.flags.writeable=False;self.fields[key]=field
    if 'dynamic_rows' in f:
     d=f['dynamic_rows'];rp=Path(path).parent/d['reference_path']
     if digest(rp)!=d['reference_sha256']:raise ValueError('Dynamic reference identity mismatch')
     ref=np.load(rp,allow_pickle=False)
     if ref.dtype!=np.float32 or ref.shape!=field.shape or not np.isfinite(ref).all() or d['strength']!=1. or d['cap_dn']!=4. or d['stride']!=4:raise ValueError('Invalid dynamic row reference or parameters')
     ref.flags.writeable=False;self.dynamic[key]=(ref,d)

  if len(self.by_key)!=len(self.index['sequences']):raise ValueError('Duplicate shared sequence')
  for r in self.by_key.values():
   if r['shape'][1:]!=[1024,1280] or len(r['parameters'])!=r['shape'][0] or r['label_scope']!='unchanged_train_roi':raise ValueError('Invalid shared RAW geometry/label scope')
   for p in r['parameters']:
    if not np.isfinite([p['offset'],p['middle_offset'],p['scale']]).all() or p['scale']<=0:raise ValueError('Invalid shared RAW normalization')
 def for_record(self,record,frame_id=None):
  r=self.by_key[record['domain']+'/'+record['sequence_id']]
  if r['raw_path']!=record['raw']['path'] or r['scene_id']!=record['scene_id'] or record['split'] not in ('train','val','test'):raise ValueError('Shared RAW source mismatch')
  f=record['frame_id'] if frame_id is None else frame_id
  if not 0<=f<len(r['parameters']):raise ValueError('Frame outside shared calibration')
  return r['parameters'][f]
 def validate_sources(self,root):
  for r in self.by_key.values():
   if digest(Path(root)/r['raw_path'])!=r['raw_sha256']:raise ValueError('Shared RAW source changed')

 def correct(self,record,raw,roi=None,input_noise=False):
  key=record['domain']+'/'+record['sequence_id']
  if key not in self.fields:return raw
  field=self.fields[key]
  if roi is not None:
   t,l,h,w=roi
   if min(t,l)<0 or min(h,w)<=0 or t+h>field.shape[0] or l+w>field.shape[1]:raise ValueError('Invalid fixed field region')
   field=field[t:t+h,l:l+w]
  if raw.shape!=field.shape:raise ValueError('Fixed field/input geometry mismatch')
  out=np.asarray(raw,dtype=np.float32)-field
  if input_noise and key in self.dynamic:
   if roi is not None:raise ValueError('Dynamic row correction requires whole RAW before cropping')
   ref,d=self.dynamic[key]
   delta=np.asarray(raw,dtype=np.float32)-ref
   row=np.median(delta[:,::d['stride']],axis=1);row-=row.mean()
   col=np.median((delta-row[:,None])[::d['stride'],:],axis=0);col-=col.mean()
   out-=d['strength']*np.clip(row[:,None]+col[None,:],-d['cap_dn'],d['cap_dn'])
  return out
