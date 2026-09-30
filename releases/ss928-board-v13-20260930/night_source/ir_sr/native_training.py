"""Explicit adaptation permission; unchanged architecture, GT, normalization and LR horizon."""
def validate_native_adaptation(previous, config, progress):
    allowed={'native_crop_sampling','native_training_mode','max_steps','max_wall_seconds','test_every','version','gpu_uuid','gpu_physical_index','cuda_memory_limit_gib'}
    changed={k for k in set(previous)|set(config) if previous.get(k)!=config.get(k)}
    if changed-allowed:raise ValueError('Unapproved native-adaptation change: '+str(sorted(changed-allowed)))
    if config.get('native_crop_sampling','center') not in ('center','uniform'):raise ValueError('Invalid sampling')
    if previous.get('native_training_mode') or config.get('native_training_mode') not in ('native_only','mixed'):
        raise ValueError('Expected frozen old-input parent and explicit native mode')
    if not (progress['step']==previous['max_steps']==8000 and config['max_steps']==12000 and config['test_every']==12000):
        raise ValueError('This experiment permits only 8000->12000')
    if previous.get('lr_schedule_steps')!=200000 or config.get('lr_schedule_steps')!=200000:
        raise ValueError('Preserve frozen LR horizon')
    if config['max_wall_seconds']<=0:raise ValueError('Positive time bound required')


def native_step(config, step):
    """Preserve the original-path prefix before alternating supervision."""
    start=config.get('native_start_step',0)
    if not isinstance(start,int) or start<0:raise ValueError('native_start_step must be a nonnegative integer')
    n,d=config.get('native_fraction_numerator',1),config.get('native_fraction_denominator',2)
    if type(n)!=int or type(d)!=int or not 0<n<d:raise ValueError('Mixed native fraction must lie strictly between zero and one')
    active=step%d>=d-n if (n,d)!=(1,2) else step%2==0
    return config.get('native_training_mode')=='native_only' or (config.get('native_training_mode')=='mixed' and step>start and active)
