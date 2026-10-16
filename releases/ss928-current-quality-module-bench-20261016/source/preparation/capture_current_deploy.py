"""Capture real tensor interfaces and invocation order for current neural modules."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import cv2
import numpy as np
import torch
from target_residual import ROOT, current_setup
from prepare_motion14 import DEST, masks
from causal_detail import GuidanceView
from tieddecode_model11 import boards
from input_anchor12 import apply
from edge_reference91 import shifted_box91, shifted_input91, PAD91, WIDTH91
from quality_shape330 import QualityShape330
from quality_control43 import QualityControl43
from raw_temporal24 import RawFeatures24
from inspect_current_deploy import roots
from background_call_deploy import BackgroundCall

BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'
READ_FILES = set()


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    path = Path(args[0]).resolve()
    if str(path).startswith(str(ROOT)+'/') and path.suffix in ['.json', '.pt']:
        READ_FILES.add(str(path))


def encode(value, arrays, path='arg', structural=False):
    if isinstance(value, torch.Tensor):
        name = 'input_%03d'%len(arrays)
        arrays[name] = value.detach().cpu().numpy().copy()
        return dict(tensor=name, shape=list(value.shape), dtype=str(arrays[name].dtype), source=path)
    if isinstance(value, (float, np.floating)) and not structural:
        name = 'input_%03d'%len(arrays)
        arrays[name] = np.full((1, 1, 1, 1), value, np.float32)
        return dict(tensor=name, shape=[1, 1, 1, 1], dtype='float32', source=path, lifted_runtime_scalar=True)
    if isinstance(value, dict):
        return dict(kind='dict', items=[[key, encode(child, arrays, path+'.'+str(key), structural or key == 'crop53')] for key, child in value.items()])
    if isinstance(value, (tuple, list)):
        return dict(kind='tuple' if isinstance(value, tuple) else 'list', items=[encode(child, arrays, path+'.'+str(index), structural) for index, child in enumerate(value)])
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return dict(kind='constant', value=value)
    raise TypeError('Unsupported runtime argument '+path+': '+str(type(value)))


def signature(value):
    if isinstance(value, torch.Tensor):
        return ['tensor', list(value.shape), str(value.dtype)]
    if isinstance(value, (float, np.floating)):
        return ['runtime_scalar_float32']
    if isinstance(value, dict):
        return ['dict', [[key, ['structural', list(child)] if key == 'crop53' else signature(child)] for key, child in value.items()]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [signature(child) for child in value]]
    if isinstance(value, np.generic):
        value = value.item()
    return ['constant', value]


def output_arrays(value, result):
    if isinstance(value, torch.Tensor):
        result['output_%03d'%len(result)] = value.detach().cpu().numpy().copy()
    elif isinstance(value, dict):
        for child in value.values():
            output_arrays(child, result)
    elif isinstance(value, (tuple, list)):
        for child in value:
            output_arrays(child, result)
    else:
        raise TypeError('Unsupported module output '+str(type(value)))


def spec_literal(spec):
    if spec.get('kind') == 'constant':
        return spec['value']
    return [spec_literal(child) for child in spec['items']]


def signature_from_spec(spec):
    if 'tensor' in spec:
        return ['runtime_scalar_float32'] if spec.get('lifted_runtime_scalar') else ['tensor', spec['shape'], 'torch.'+spec['dtype']]
    kind = spec['kind']
    if kind == 'dict':
        return ['dict', [[key, ['structural', spec_literal(child)] if key == 'crop53' else signature_from_spec(child)] for key, child in spec['items']]]
    if kind in ['tuple', 'list']:
        return [kind, [signature_from_spec(child) for child in spec['items']]]
    return ['constant', spec['value']]


class Recorder:
    def __init__(self, scene, refresh_frame=None):
        self.scene, self.split, self.frame = scene, None, None
        self.refresh_frame, self.refreshed = refresh_frame, set()
        self.graphs, self.by_signature, self.calls, self.handles = [], {}, [], []
        self.module_ids, self.seen_modules = {}, {}
        manifest = PACKAGE/'models'/scene/'capture_manifest.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            self.graphs = previous['graphs']
            self.module_ids = {item['path']: item['id'] for item in previous['neural_modules']}
            self.seen_modules = {item['id']: item for item in previous['neural_modules']}
            for graph in self.graphs:
                spec = graph['argument_spec']
                key = json.dumps([graph['module_id'], signature_from_spec(spec['args']), signature_from_spec(spec['kwargs'])], sort_keys=True)
                self.by_signature[key] = graph

    def attach(self, modules):
        for path, module in modules:
            if path not in self.module_ids:
                self.module_ids[path] = 'module_%02d'%len(self.module_ids)
            module_id = self.module_ids[path]
            self.seen_modules[module_id] = dict(id=module_id, path=path, type=type(module).__module__+'.'+type(module).__name__)
            def hook(inner, args, kwargs, output, module_id=module_id, path=path):
                key = json.dumps([module_id, signature(args), signature(kwargs)], sort_keys=True)
                if key not in self.by_signature:
                    graph_id = 'graph_%03d'%len(self.graphs)
                    folder = PACKAGE/'models'/self.scene/graph_id
                    folder.mkdir(parents=True, exist_ok=True)
                    arrays = {}
                    spec = dict(args=encode(args, arrays, 'args'), kwargs=encode(kwargs, arrays, 'kwargs'))
                    graph = dict(id=graph_id, scene=self.scene, module_id=module_id, module_path=path,
                        module_type=type(inner).__module__+'.'+type(inner).__name__, folder=str(folder.relative_to(PACKAGE)),
                        argument_spec=spec, cases={}, ONNX_exported=False)
                    self.by_signature[key] = graph
                    self.graphs.append(graph)
                    # Keep weights independent of recorder hooks and Python closures.
                    weights = folder.parent/(module_id+'.pt')
                    if not weights.exists():
                        torch.save(dict(module_type=graph['module_type'], module_path=path,
                            state_dict={name: value.detach().cpu() for name, value in inner.state_dict().items()}), weights)
                    (folder/'argument_spec.json').write_text(json.dumps(spec, indent=2))
                graph = self.by_signature[key]
                has_fp64 = '"float64"' in json.dumps(graph['argument_spec'])
                refresh = has_fp64 and self.frame == self.refresh_frame and (graph['id'], self.split) not in self.refreshed
                if self.split not in graph['cases'] or refresh:
                    folder = PACKAGE/graph['folder']
                    arrays, outputs = {}, {}
                    encode(args, arrays, 'args')
                    encode(kwargs, arrays, 'kwargs')
                    output_arrays(output, outputs)
                    np.savez_compressed(folder/(self.split+'_inputs.npz'), **arrays)
                    np.savez_compressed(folder/(self.split+'_pytorch_output.npz'), **outputs)
                    graph['cases'][self.split] = dict(frame=self.frame, split=self.split, source='original RAW path; no GT in inputs')
                    if refresh:
                        self.refreshed.add((graph['id'], self.split))
                self.calls.append(graph['id'])
            self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def save(self, split, frame):
        folder = PACKAGE/'workloads'/self.scene
        folder.mkdir(parents=True, exist_ok=True)
        data = dict(scene=self.scene, split=split, frame=frame, calls=self.calls,
            counts=dict(Counter(self.calls)), input_preparation='PC original pipeline, outside component NPU timing',
            full_pipeline=False, CPU_state_and_fusion_not_executed_on_board=True)
        (folder/(split+'_frame_%03d.json'%frame)).write_text(json.dumps(data, indent=2))
        self.calls = []

    def flush(self):
        folder = PACKAGE/'models'/self.scene
        folder.mkdir(parents=True, exist_ok=True)
        called = {graph['module_id'] for graph in self.graphs}
        method_covered = {value['path'][:-len('.background')] for key, value in self.seen_modules.items() if key in called and value['path'].endswith('.background')}
        result = dict(scene=self.scene, neural_modules=list(self.seen_modules.values()), graphs=self.graphs,
            modules_used_via_background_method=sorted(method_covered),
            modules_without_observed_call=[value for key, value in self.seen_modules.items() if key not in called and value['path'] not in method_covered],
            frames_captured=max(20, (self.refresh_frame or 0)+1), calibration_split='train', smoke_split='val',
            fixed_SR_geometry='current continuous validation geometry; training RAW supplies calibration values')
        (folder/'capture_manifest.json').write_text(json.dumps(result, indent=2))
        print('DEPLOY_CAPTURE_SUMMARY', self.scene, len(self.graphs), 'graphs', 'uncalled', len(result['modules_without_observed_call']), flush=True)


def capture(scene, refresh_frame=None):
    recorder = Recorder(scene, refresh_frame)
    kind, config, df, box, count, base, original, profile = current_setup(scene)
    del original
    val_meta = json.loads((DEST/scene/'val_continuous'/'capture_00'/'complete.json').read_text())
    for split, cache_split in [('val', 'val_continuous'), ('train', 'train')]:
        meta = json.loads((DEST/scene/cache_split/'capture_00'/'complete.json').read_text())
        source = df(config, split)
        apply(source, scene)
        data = GuidanceView(source, list(range(9-count, 9)))
        parent, model = boards(scene, base, count, kind, box(val_meta['row']), 500)
        del parent
        edge_model = None
        if scene == 'weather_heavy_c32':
            parent, edge_model = boards(scene, base, count, kind, shifted_box91(box(val_meta['row']), val_meta['row']['eval_crop_tlhw'][3]), 500)
            del parent
        controller = QualityShape330(332) if scene == 'weather_heavy_c32' else QualityControl43(scene)
        modules = [('SR.primary', model)] + ([('SR.edge', edge_model)] if edge_model is not None else []) + roots(controller)
        for module_path, neural in list(modules):
            if type(neural).__name__ == 'LearnedBackground243':
                method = BackgroundCall(neural)
                def invoke(*values, _method=method, **keywords):
                    return _method(*values, **keywords)
                neural.background = invoke
                modules.append((module_path+'.background', method))
        recorder.attach(modules)
        raw = RawFeatures24(scene) if scene == 'night_special' else None
        recorder.split = split
        row = meta['row']
        requested = max(20, (refresh_frame or 0)+1)
        frames = meta['frames'][:requested]
        if len(frames) != requested or frames != list(range(requested)):
            raise ValueError('Consecutive causal frames required')
        with torch.inference_mode():
            for index, frame in enumerate(frames):
                recorder.frame = frame
                rr = dict(row, frame_id=frame)
                data.current_record, data.current_crop, data.cache = rr, (0, 0, 1024, 1280), {}
                context, _ = data.context_for(rr)
                context = context[None].cuda().half()
                inputs = data.inputs(profile)[None].cuda()[:, -model.source_count:].half()
                full = model(inputs, context).float()
                activity, occupancy, _ = masks(data._nine_without_field().clone())
                moving = torch.from_numpy(activity)[None, None].cuda().half()
                current = inputs[:, -1:]
                past = torch.median(inputs[:, :-1].float(), dim=1, keepdim=True).values.half()
                variants = dict(current_only=current.expand_as(inputs).contiguous(),
                    gated_history=inputs*(1-moving)+current*moving,
                    past_background=past.expand_as(inputs).contiguous())
                fields = {'baseline': full[..., ::3, ::3].half().float()}
                for name, variant in variants.items():
                    output = model(variant, context).float()
                    fields[name] = output[..., ::3, ::3].half().float()
                    if name == 'current_only':
                        current_full = output
                fields['activity'] = torch.from_numpy(activity)[None, None].cuda().half().float()
                fields['occupancy'] = torch.from_numpy(occupancy)[None, None].cuda().half().float()
                if scene == 'weather_heavy_c32':
                    plain = data._nine_without_field()[-1:][None].cuda().half().expand_as(inputs).contiguous()
                    fields['plain_current116'] = model(plain, context).float()[..., ::3, ::3].half().float()
                    shifted = edge_model(shifted_input91(variants['current_only']), context)
                    fields['shifted_current91'] = shifted[..., ::3, 3*PAD91:3*(PAD91+WIDTH91):3].half().float()
                    native = controller.native(fields, row, frame, initial=index == 0)
                    after = controller.full(full, native, current_full, shifted.float(), fields)
                else:
                    if raw is not None:
                        fields['raw24'] = raw.for_record(row, frame, (0, 0, 1024, 1280))
                    native = controller.native(fields, initial=index == 0)
                    from quality_control43 import lift_native
                    after = lift_native(full, native)
                recorder.save(split, frame)
                if index % 5 == 0 or index+1 == len(frames):
                    recorder.flush()
                    print('DEPLOY_CAPTURE_FRAME', scene, split, frame, flush=True)
                del after, native, output, current_full, full, inputs, context, fields, variants
        recorder.detach()
        del data, model, edge_model, controller, modules
        gc.collect()
        torch.cuda.empty_cache()
    recorder.flush()
    evidence = PACKAGE/'source'
    runtime = evidence/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    for module in list(sys.modules.values()):
        source_path = getattr(module, '__file__', None)
        if source_path and str(source_path).startswith(str(ROOT/'runtime')+'/') and str(source_path).endswith('.py'):
            shutil.copy2(source_path, runtime/Path(source_path).name)
    dependencies = []
    for source_path in sorted(READ_FILES):
        original_path = Path(source_path)
        if not original_path.is_file():
            continue
        relative = original_path.relative_to(ROOT)
        target = evidence/'run_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(original_path, target)
        dependencies.append(dict(original=str(original_path), packaged=str(target.relative_to(PACKAGE))))
    (evidence/(scene+'_dependencies.json')).write_text(json.dumps(dependencies, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['weather_heavy_c32', 'weather_heavy', 'night_special'], required=True)
    parser.add_argument('--refresh-frame', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    sys.addaudithook(audit)
    capture(args.scene, args.refresh_frame)
