"""Restore the declared historical training-ROI normalization policy; no new statistics."""
import types

def apply(source,scene):
    base=source.base;original=base.normalization_for
    if scene!='weather_heavy_c32':return dict(changed=False,scene=scene)
    normalizer=base.sequence_normalization
    if not normalizer.regional:raise ValueError('Frozen regional index required')
    def normalization_for(self,record,frame_id=None):
        if record.get('scene_id')!=scene or record['split']=='train':return original(record,frame_id)
        if record.get('evaluation_scope')!='spatial_development':raise ValueError('Historical anchor requires same-capture spatial split')
        prefix=record['domain']+'/'+record['sequence_id']+'|train|'
        found=[r for k,r in normalizer.by_key.items() if k.startswith(prefix)]
        if len(found)!=1:raise ValueError('Unique frozen same-capture training calibration required')
        row=found[0]
        if row['raw_path']!=record['raw']['path'] or row['split']!='train':raise ValueError('Calibration source mismatch')
        frame=record['frame_id'] if frame_id is None else frame_id
        if not 0<=frame<row['frames']:raise ValueError('Frame outside existing training calibration')
        return dict(row['parameters'][frame])
    base.normalization_for=types.MethodType(normalization_for,base)
    return dict(changed=True,scene=scene,policy='Existing frozen same-capture training-ROI parameters for val/test, training unchanged',
        new_statistics_estimated=False,GT_used=False,context_scope_changed=False,old_reference_and_weights_changed=False)
