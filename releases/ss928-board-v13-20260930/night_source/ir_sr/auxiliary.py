"""Training-only RAW supervision and sampled CUDA module timing."""
import json
from pathlib import Path

import torch

from ir_sr.model import RT4KSRB0, architecture_options
from ir_sr.training import sha


class FeatureAffine(torch.nn.Module):
    """Original downsampled training path only; native inference omits this module."""
    def __init__(self,channels):
        super().__init__()
        self.gain=torch.nn.Parameter(torch.ones(1,channels,1,1))
        self.bias=torch.nn.Parameter(torch.zeros(1,channels,1,1))
    def forward(self,value):return value*self.gain.to(value.dtype)+self.bias.to(value.dtype)


def training_model(config):
    weight = config.get('auxiliary_raw_weight', 0)
    if weight < 0 or bool(weight) != bool(config.get('middle_gt_root')):
        raise ValueError('Auxiliary loss and target cache must be enabled together')
    model = RT4KSRB0(config['channels'], config['blocks'], auxiliary_raw=weight > 0 and not config.get('temporal_bottleneck',False), global_reference=config.get('global_reference', False), **architecture_options(config))

    if config.get('reference_tone_weight',0)<0:raise ValueError('Invalid reference supervision weight')
    if config.get('reference_tone_weight',0)>0:
        if not config.get('global_reference'):raise ValueError('Reference supervision requires reference model')
        model.reference_tone_head=torch.nn.Conv2d(12,1,1)
        torch.nn.init.zeros_(model.reference_tone_head.weight)
        torch.nn.init.zeros_(model.reference_tone_head.bias)
    if config.get("scale_training_adapter"):
        model.original_scale_adapter=torch.nn.ModuleList([FeatureAffine(config["channels"]) for _ in range(3)])
    return model


def reference_tone_loss(model, context, target, weight):
    if not weight:return next(model.parameters()).new_zeros((),dtype=torch.float32)
    if target is None or target.shape!=(context.shape[0],1,64,64):raise ValueError('Reference GT shape mismatch')
    prediction=model.reference_tone_head(model.global_reference.encode(context)).float()
    return (prediction-target.float()).abs().mean()


def loss_terms(model, raw, target, middle, weight, raw_loss_multiplier=None, context=None, context_box=None, display_downsample=1, return_prediction=False, return_motion_gate=False):
    context_args = {'context': context, 'context_box': context_box} if context is not None else {}
    if hasattr(model,'original_scale_adapter'):
        context_args['input_downsample']=3 if display_downsample==1 else 1
    if weight:
        if middle is None or middle.shape != raw[:, -1:].shape:
            raise ValueError('Auxiliary target must match the input RAW grid')
        outputs = model(raw, return_auxiliary=True, return_motion_gate=return_motion_gate, **context_args)
        prediction, auxiliary = outputs[:2]
        motion_gate = outputs[2] if return_motion_gate else None
        raw_error = (auxiliary.float() - middle.float()).abs()
        if raw_loss_multiplier is not None:
            multiplier = raw_loss_multiplier.to(device=raw.device, dtype=torch.float32).reshape(-1,1,1,1)
            if multiplier.shape[0] != raw.shape[0] or not torch.isfinite(multiplier).all() or (multiplier <= 0).any():
                raise ValueError('Invalid per-sample RAW loss scale')
            raw_error = raw_error * multiplier
        raw_l1 = raw_error.mean()
    else:
        prediction = model(raw, **context_args)
        raw_l1 = raw.new_zeros((), dtype=torch.float32)
    if display_downsample not in (1,3):raise ValueError('Unsupported display supervision scale')
    prediction=prediction.float()
    if display_downsample==3:prediction=torch.nn.functional.avg_pool2d(prediction,3)
    if prediction.shape!=target.shape:raise ValueError('Display target shape mismatch')
    display_l1 = (prediction - target.float()).abs().mean()
    terms = (display_l1 + weight * raw_l1, display_l1, raw_l1)
    if return_motion_gate:
        if not return_prediction:raise ValueError('Gate supervision requires prediction')
        return (*terms,prediction,motion_gate)
    return (*terms,prediction) if return_prediction else terms


def validate_middle_lock(config, dataset):
    if not config.get('middle_gt_root'):
        return None
    root = Path(config['middle_gt_root'])
    path = root / 'index.json'
    assert sha(path) == config['middle_gt_index_sha256'], 'Middle target manifest changed'
    index = json.loads(path.read_text())
    assert index['status'] == 'complete' and index['source_after_verified']
    data = Path(config['data_root'])
    assert index['train_split_sha256'] == sha(data / 'manifests/split_s1/train.txt')
    assert index['pairs_sha256'] == sha(data / 'manifests/dataset_d1/pairs.jsonl')
    selected = {r['domain'] + '/' + r['sequence_id'] for r in dataset.records}
    actual = set()
    for row in index['sequences']:
        if row['key'] in selected:
            assert sha(root / row['path']) == row['sha256'], 'Middle target content changed'
            assert sha(data / row['source_raw_path']) == row['source_decoded_sha256'], 'Decoded RAW changed'
            actual.add(row['key'])
    assert actual == selected
    return sha(path)


class ModuleTimer:
    """CUDA event intervals on sampled training steps, with no per-module synchronization."""
    def __init__(self, model):
        self.active = False
        self.events = {}
        self.handles = []
        for name in ('temporal_pre', 'motion_pre', 'down', 'head', 'body', 'tail', 'upsample', 'auxiliary_raw', 'global_reference'):
            module = getattr(model, name, None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(lambda m, a, n=name: self.begin(n)))
                self.handles.append(module.register_forward_hook(lambda m, a, o, n=name: self.end(n)))

    def begin(self, name):
        if self.active:
            event = torch.cuda.Event(enable_timing=True); event.record()
            self.events[name] = [event, None]

    def end(self, name):
        if self.active:
            event = torch.cuda.Event(enable_timing=True); event.record()
            self.events[name][1] = event

    def start_step(self, active):
        self.active = active
        self.events = {}

    def finish(self):
        if not self.active:
            return None
        torch.cuda.synchronize()
        result = {name: start.elapsed_time(end) for name, (start, end) in self.events.items()}
        self.active = False
        return result
