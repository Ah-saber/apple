"""Export recorded tensor modules with runtime scalar inputs preserved."""
import argparse
import copy
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import sys
import traceback
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from target_residual import current_setup
from prepare_motion14 import DEST
from tieddecode_model11 import boards
from edge_reference91 import shifted_box91
from quality_shape330 import QualityShape330
from quality_control43 import QualityControl43
from inspect_current_deploy import roots
from background_call_deploy import BackgroundCall

PACKAGE = Path('/tmp/rawir-current-deploy-20261016/package')


def rebuild(spec, tensors):
    if 'tensor' in spec:
        return tensors[spec['tensor']]
    if spec['kind'] == 'constant':
        return spec['value']
    if spec['kind'] == 'dict':
        return {key: rebuild(child, tensors) for key, child in spec['items']}
    values = [rebuild(child, tensors) for child in spec['items']]
    return tuple(values) if spec['kind'] == 'tuple' else values


def tensors_in(value):
    if isinstance(value, torch.Tensor):
        return [value]
    children = value.values() if isinstance(value, dict) else value
    result = []
    for child in children:
        result.extend(tensors_in(child))
    return result


class FixedCall(torch.nn.Module):
    def __init__(self, inner, spec, names):
        super().__init__()
        self.inner, self.spec, self.names = inner, spec, names

    def forward(self, *values):
        tensors = dict(zip(self.names, values))
        result = self.inner(*rebuild(self.spec['args'], tensors), **rebuild(self.spec['kwargs'], tensors))
        return tuple(tensors_in(result))


def materialize_aliases(graph):
    known = {value.name: value for value in graph.graph.initializer}
    outputs = {value.name for value in graph.graph.output}
    changed = True
    while changed:
        changed, kept = False, []
        for node in graph.graph.node:
            if node.op_type == 'Identity' and node.input[0] in known and node.output[0] not in outputs:
                tensor = copy.deepcopy(known[node.input[0]])
                tensor.name = node.output[0]
                graph.graph.initializer.append(tensor)
                known[tensor.name] = tensor
                changed = True
            else:
                kept.append(node)
        del graph.graph.node[:]
        graph.graph.node.extend(kept)


def describe(value):
    tensor = value.type.tensor_type
    return dict(name=value.name, dtype=onnx.TensorProto.DataType.Name(tensor.elem_type),
        shape=[dimension.dim_value for dimension in tensor.shape.dim])


def construct(scene):
    kind, config, df, box, count, base, original, profile = current_setup(scene)
    del original
    meta = json.loads((DEST/scene/'val_continuous'/'capture_00'/'complete.json').read_text())
    parent, model = boards(scene, base, count, kind, box(meta['row']), 500)
    del parent
    modules = [('SR.primary', model)]
    if scene == 'weather_heavy_c32':
        parent, edge = boards(scene, base, count, kind, shifted_box91(box(meta['row']), meta['row']['eval_crop_tlhw'][3]), 500)
        del parent
        modules.append(('SR.edge', edge))
    controller = QualityShape330(332) if scene == 'weather_heavy_c32' else QualityControl43(scene)
    modules += roots(controller)
    for path, module in list(modules):
        if type(module).__name__ == 'LearnedBackground243':
            modules.append((path+'.background', BackgroundCall(module)))
    return dict(modules), controller


def export(scene):
    folder = PACKAGE/'models'/scene
    manifest = json.loads((folder/'capture_manifest.json').read_text())
    modules, controller = construct(scene)
    errors = []
    for item in manifest['graphs']:
        target = PACKAGE/item['folder']
        if (target/'manifest.json').exists():
            continue
        try:
            module = copy.deepcopy(modules[item['module_path']])
            for child in module.modules():
                child._forward_hooks.clear()
                child._forward_pre_hooks.clear()
                if hasattr(child, 'records'):
                    child.records = []
            state = torch.load(folder/(item['module_id']+'.pt'), map_location='cpu', weights_only=False)
            module.load_state_dict(state['state_dict'])
            module = module.cuda().eval().requires_grad_(False)
            case = 'val' if 'val' in item['cases'] else 'train'
            source_path = target/(case+'_inputs.npz')
            if not source_path.exists():
                source_path = PACKAGE.parent/'capture_archive'/target.relative_to(PACKAGE)/(case+'_inputs.npz')
            source = np.load(source_path)
            names = list(source.files)
            precision_changes = [dict(input=name, source_dtype='float64', deployment_dtype='float32') for name in names if source[name].dtype == np.float64]
            arrays = {name: source[name].astype(np.float32) if source[name].dtype == np.float64 else source[name] for name in names}
            wrapper = FixedCall(module, item['argument_spec'], names).cuda().eval()
            values = tuple(torch.from_numpy(arrays[name]).cuda() for name in names)
            with torch.inference_mode():
                direct = [value.detach().cpu().numpy() for value in wrapper(*values)]
                output_names = ['output_%03d'%index for index in range(len(direct))]
                path = target/'model.onnx'
                torch.onnx.export(wrapper, values, str(path), input_names=names, output_names=output_names,
                    opset_version=17, dynamo=False, do_constant_folding=True)
            graph = onnx.load(path)
            materialize_aliases(graph)
            onnx.checker.check_model(graph)
            onnx.save(graph, path)
            input_names = [value.name for value in graph.graph.input]
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            pc = session.run(None, {name: arrays[name] for name in input_names})
            # Fixed-shape deployment outputs must not retain exporter symbolic dimensions.
            for value, actual in zip(graph.graph.output, pc):
                del value.type.tensor_type.shape.dim[:]
                for dimension in actual.shape:
                    value.type.tensor_type.shape.dim.add().dim_value = int(dimension)
            onnx.checker.check_model(graph)
            onnx.save(graph, path)
            comparisons = []
            for index, (actual, expected) in enumerate(zip(pc, direct)):
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError('Output shape or finite-value check failed')
                difference = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
                comparisons.append(dict(output=output_names[index], MAE=float(difference.mean()),
                    max_absolute=float(difference.max()), P95_absolute=float(np.percentile(difference, 95))))
            np.savez_compressed(target/'pc_output.npz', **dict(zip(output_names, pc)))
            # Store the exact graph inputs; traced but unused Python arguments are excluded.
            for case_name in item['cases']:
                original_path = target/(case_name+'_inputs.npz')
                if not original_path.exists():
                    original_path = PACKAGE.parent/'capture_archive'/target.relative_to(PACKAGE)/(case_name+'_inputs.npz')
                original = np.load(original_path)
                np.savez_compressed(target/(case_name+'_graph_inputs.npz'), **{name: original[name].astype(np.float32) if original[name].dtype == np.float64 else original[name] for name in input_names})
            result = dict(**item, ONNX_file='model.onnx', inputs=[describe(value) for value in graph.graph.input],
                outputs=[describe(value) for value in graph.graph.output], operators=dict(Counter(node.op_type for node in graph.graph.node)),
                checker_passed=True, PC_runtime='onnxruntime CPU', PC_case=case,
                PyTorch_vs_PC=comparisons, SDK_compiled=False, NPU_measured=False,
                full_pipeline=False, CPU_tracking_and_fusion_outside_graph=True,
                runtime_noise_scalars_are_explicit_inputs=True,
                boundary_precision_changes=precision_changes,
                argument_spec_note='Original research call layout; deployed input precision is given by inputs')
            observed_path = target/(case+'_pytorch_output.npz')
            if precision_changes and observed_path.exists():
                observed = np.load(observed_path)
                precision_only = []
                for index, name in enumerate(output_names):
                    difference = np.abs(direct[index].astype(np.float32)-observed[name].astype(np.float32))
                    precision_only.append(dict(output=name, MAE=float(difference.mean()), max_absolute=float(difference.max())))
                result['GPU_boundary_conversion_comparison'] = precision_only
            result['ONNX_exported'] = True
            (target/'manifest.json').write_text(json.dumps(result, indent=2))
            print('CURRENT_EXPORT_DONE', scene, item['id'], item['module_type'], flush=True)
            del module, wrapper, values, direct, graph, session, pc, arrays, source, original
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            error = traceback.format_exc()
            (target/'export_error.txt').write_text(error)
            errors.append(dict(graph=item['id'], error=error))
            print('CURRENT_EXPORT_FAILED', scene, item['id'], error, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    summary = dict(scene=scene, total_graphs=len(manifest['graphs']),
        exported=sum((PACKAGE/item['folder']/'manifest.json').exists() for item in manifest['graphs']), errors=errors,
        modules_without_observed_call=manifest['modules_without_observed_call'])
    (folder/'export_summary.json').write_text(json.dumps(summary, indent=2))
    runtime = PACKAGE/'source'/'runtime'
    for imported in list(sys.modules.values()):
        source_path = getattr(imported, '__file__', None)
        if source_path and '/SS928-ROUND5-20261007/runtime/' in str(source_path) and str(source_path).endswith('.py'):
            shutil.copy2(source_path, runtime/Path(source_path).name)
    print('CURRENT_EXPORT_SUMMARY', json.dumps(summary), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['weather_heavy_c32', 'weather_heavy', 'night_special'], required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    export(args.scene)
