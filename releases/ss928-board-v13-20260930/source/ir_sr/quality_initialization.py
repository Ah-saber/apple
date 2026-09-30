"""Strict warm initialization for paired absolute-reference ablation. Adam/RNG are fresh."""
from pathlib import Path
import torch
from ir_sr.training import sha
from ir_sr.model import architecture_options

def training_path_steps(path, expected_sha=None, seen=None):
    """Count ancestral optimizer steps, including nested fresh-Adam warm starts."""
    path=Path(path).resolve();seen=set() if seen is None else set(seen)
    if path in seen:raise ValueError('Cyclic initialization ancestry')
    seen.add(path)
    if expected_sha is not None and sha(path)!=expected_sha:raise ValueError('Ancestral weight SHA mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False)
    steps=state['progress']['step'];config=state['config']
    if type(steps)!=int or steps<0:raise ValueError('Invalid ancestral step count')
    if config.get('initialization_checkpoint'):
        steps+=training_path_steps(config['initialization_checkpoint'],config['initialization_checkpoint_sha256'],seen)
    return steps


def validate_training_budget(config):
    parent=training_path_steps(config['initialization_checkpoint'],config['initialization_checkpoint_sha256']) if config.get('initialization_checkpoint') else 0
    if type(config['max_steps'])!=int or config['max_steps']<=0 or parent+config['max_steps']>20000:
        raise ValueError('Training path exceeds authorized 20000 steps')
    return dict(parent_steps=parent,stage_max_steps=config['max_steps'],total_max_steps=parent+config['max_steps'])


def initialize_quality_model(model, config):
    validate_training_budget(config)
    path=config['initialization_checkpoint']
    if sha(path)!=config['initialization_checkpoint_sha256']:raise ValueError('Parent weight SHA mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False);previous=source['config']
    # Only one architecture option differs. Frozen input, labels and selected scenes remain identical.
    for key in ('channels','blocks','auxiliary_raw_weight','global_reference','scene_ids','sequence_normalization_index','sequence_normalization_sha256','middle_gt_root','middle_gt_index_sha256','data_root'):
        if config.get(key)!=previous.get(key):raise ValueError('Warm-start invariant changed: '+key)
    for key,value in architecture_options(previous).items():
        if key not in ('input_frames','temporal_bottleneck','motion_gate','trajectory_gate') and architecture_options(config)[key]!=value:raise ValueError('Architecture changed: '+key)
    state=dict(source['model']);key='global_reference.encoder.0.weight'
    if config.get('absolute_raw_reference'):
        if state[key].shape[1]!=1:raise ValueError('Expected one-channel parent')
        expanded=torch.zeros_like(model.state_dict()[key],device='cpu');expanded[:,:1]=state[key];state[key]=expanded
    if config.get('temporal_bottleneck', False):
        if previous.get('input_frames',1)!=1 or previous.get('temporal_bottleneck',False):
            raise ValueError('Bottleneck warm start requires a one-frame parent')
        auxiliary_keys={key for key in state if key.startswith('auxiliary_raw.')}
        if auxiliary_keys != {'auxiliary_raw.0.weight','auxiliary_raw.0.bias'}:
            raise ValueError('Expected parent auxiliary RAW head')
        state={key:value for key,value in state.items() if key not in auxiliary_keys}
    key='head.0.weight'
    if state[key].shape != model.state_dict()[key].shape:
        if previous.get('input_frames',1)!=1 or config.get('input_frames')!=3:raise ValueError('Only 1-to-3-frame initialization supported')
        expanded=torch.zeros_like(model.state_dict()[key],device='cpu');expanded[:,-state[key].shape[1]:]=state[key];state[key]=expanded
    for key,value in model.state_dict().items():
        if key not in state and key.startswith(('reference_tone_head.','original_scale_adapter.')):
            state[key]=value.detach().cpu().clone()
    if config.get('temporal_bottleneck', False):
        missing,unexpected=model.load_state_dict(state,strict=False)
        expected={key for key in model.state_dict() if key.startswith(('temporal_pre.','motion_pre.'))}
        if set(missing)!=expected or unexpected:
            raise ValueError('Unexpected bottleneck initialization keys')
    else:
        model.load_state_dict(state,strict=True)
    return dict(checkpoint=path,sha256=sha(path),parent_steps=source['progress']['step'],additional_steps=config['max_steps'],optimizer='Fresh Adam for both arms',reference_new_channel=('Zero-initialized temporal RAW correction; initial display output unchanged' if config.get('temporal_bottleneck', False) else 'Zero-initialized input weights; initial output unchanged'),initialization='model weights only')
