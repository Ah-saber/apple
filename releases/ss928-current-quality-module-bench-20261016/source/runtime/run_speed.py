"""Pending full GPU timing and exact-output candidate checks; no NPU claims."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

R4 = Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-LOCAL-ROUND4-20261001')
sys.path.insert(0, str(R4/'runtime'))
import run_structural as frozen
from candidate_ops import GroupedSeparableCopyOutput


def cases(backups):
    for group, scenes in [('day', ['day_normal']), ('light', ['weather_light', 'weather_medium']),
                          ('heavy', ['weather_heavy', 'weather_heavy_c32'])]:
        compact, base, config, _, _, df, box_for = frozen.load_group(group)
        data = df(config, 'test')
        for scene in scenes:
            box = box_for(next(r for r in data.records if r['scene_id'] == scene)).cuda()
            yield scene, frozen.BoardHalfOptimized(base, box).cuda().half().eval(), frozen.V13/'deployment_corrected'/scene
            if backups and group != 'day':
                yield scene+'_quarter', frozen.BoardCompact(compact, box).cuda().half().eval(), frozen.V13/'deployment_corrected'/scene
    for scene in ['ordinary', 'special']:
        yield 'night_'+scene, frozen.NightFixed(frozen.night_source(scene), scene).cuda().eval(), frozen.V13/'night_calibration_final'/('night_'+scene)


def make_model(original, blocks, output):
    model = copy.deepcopy(original)
    sampling = frozen.replace_samples(model, blocks) if blocks else []
    if not blocks:
        frozen.replace_fixed_samples(model)
    attr = 'rows' if isinstance(model, frozen.BoardHalfOptimized) else 'output'
    factor = 4 if isinstance(model, frozen.BoardCompact) else 2
    if output == 'rows4':
        finish = frozen.GroupRows(4, factor)
    elif output == 'copy':
        finish = frozen.ChannelCopyOutput(factor)
    else:
        finish = GroupedSeparableCopyOutput(factor, output.removeprefix('grouped_'), grouped=output.startswith('grouped_'))
    setattr(model, attr, finish.cuda().half())
    return model.eval(), sampling


def timed(model, inputs, warm, steps):
    # Each measured call produces the full display tensor, then synchronizes.
    # GPU event spans and wall time are reported separately, both are GPU evidence.
    with torch.inference_mode():
        for i in range(warm):
            result = model(*inputs[i % len(inputs)])
        torch.cuda.synchronize()
        gpu, wall = [], []
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        for i in range(steps):
            before = time.perf_counter_ns()
            start.record()
            result = model(*inputs[i % len(inputs)])
            finish.record()
            finish.synchronize()
            wall.append((time.perf_counter_ns()-before)/1e6)
            gpu.append(start.elapsed_time(finish))
    assert result.shape == (1, 1, 3072, 3840) and result.dtype == torch.float16
    return {name: {'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95)),
                   'samples_ms': values} for name, values in [('gpu_event', gpu), ('full_synchronized_wall', wall)]}



measure = timed


def export_selected(model, args, path):
    import onnx
    import onnxruntime as ort
    torch.onnx.export(model, args, str(path), input_names=['nine_raw', 'reference_thumb'],
                      output_names=['display_gray'], opset_version=17, dynamo=False)
    graph = onnx.load(str(path))
    aliases = frozen.materialize_conv_aliases(graph)
    gate = frozen.inspect(graph)
    if not gate['passes_observed_pattern_gate']:
        raise ValueError(gate)
    onnx.save(graph, str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    feeds = {v.name: args[i].cpu().numpy().astype(np.float16 if v.type == 'tensor(float16)' else np.float32)
             for i, v in enumerate(session.get_inputs())}
    actual = session.run(None, feeds)[0]
    with torch.inference_mode():
        expected = model(*args).cpu().numpy()
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid exported output')
    diff = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
    return {'path': str(path), 'gate': gate, 'aliases': aliases, **frozen.graph_meta(graph),
            'ORT_last_input': {'max_gray': float(diff.max()), 'mean_gray': float(diff.mean())}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backups', action='store_true')
    p.add_argument('--export', action='store_true')
    p.add_argument('--scenes', nargs='+')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warm', type=int, default=50)
    a = p.parse_args()
    assert a.steps > 0 and a.rounds > 0 and a.warm >= 0
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    variants = [('control', None, 'rows4'), ('block8_copy', 8, 'copy'),
                ('block8_separable_width', 8, 'width'), ('block8_grouped_width', 8, 'grouped_width'),
                ('block8_grouped_height', 8, 'grouped_height'), ('block16_grouped_width', 16, 'grouped_width')]
    records = {'NPU_measured': False, 'SDK_compiled': False, 'hash_checks': False,
               'timing_includes_full_output': True, 'input_transfer_timed': False,
               'host_postprocessing_timed': False, 'gpu': torch.cuda.get_device_name(0),
               'steps_per_round': a.steps, 'rounds': a.rounds, 'warm_per_model_round': a.warm,
               'control': 'same frozen weights and external FP16 full output; R3 weighted sampling and rows4',
               'scenes': {}}
    rng = np.random.default_rng(1002)
    for scene, original, directory in cases(a.backups):
        if a.scenes and scene not in a.scenes:
            del original
            continue
        dest = a.out/scene
        dest.mkdir()
        inputs, files = [], []
        for path in sorted(directory.glob('test_*.npz')):
            sample = np.load(path, allow_pickle=False)
            args = tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw', 'reference_thumb'])
            if not scene.startswith('night_'):
                args = tuple(v.half() for v in args)
            inputs.append(args)
            files.append({'path': str(path)})
        assert len(inputs) == 12
        control, _ = make_model(original, None, 'rows4')
        entry = {'inputs': files, 'variants': {}, 'round_orders': []}
        for label, blocks, output in variants:
            model, sampling = make_model(original, blocks, output)
            differences = []
            with torch.inference_mode():
                for args in inputs:
                    expected, actual = control(*args), model(*args)
                    assert actual.shape == (1, 1, 3072, 3840) and actual.dtype == torch.float16
                    assert torch.isfinite(actual).all()
                    difference = frozen.delta(actual, expected)
                    assert difference['max_gray'] <= 1, (scene, label, difference)
                    differences.append(difference)
            entry['variants'][label] = {'sampling': sampling, 'controls': differences, 'timing_rounds': []}
            del model, expected, actual
        del control
        for round_id in range(a.rounds):
            order = rng.permutation(len(variants)).tolist()
            entry['round_orders'].append([variants[i][0] for i in order])
            for index in order:
                label, blocks, output = variants[index]
                model, _ = make_model(original, blocks, output)
                result = timed(model, inputs, a.warm, a.steps)
                entry['variants'][label]['timing_rounds'].append(result)
                print('FULL_GPU_TIMING', scene, round_id, label, result['full_synchronized_wall']['mean_ms'], flush=True)
                del model
        for label, value in entry['variants'].items():
            value['timing'] = {}
            for method in ['gpu_event', 'full_synchronized_wall']:
                values = [v for r in value['timing_rounds'] for v in r[method]['samples_ms']]
                value['timing'][method] = {'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95))}
        base = entry['variants']['control']['timing']
        for label, value in entry['variants'].items():
            value['relative_to_control'] = {
                method: {key: 100*(1-value['timing'][method][key]/base[method][key])
                         for key in ['mean_ms', 'p95_ms']} for method in base}
        winner = min(entry['variants'], key=lambda label: entry['variants'][label]['timing']['full_synchronized_wall']['mean_ms'])
        entry['gpu_fastest_only'] = winner
        if a.export and winner != 'control':
            entry['exports'] = {}
            for label in ['control', winner]:
                _, blocks, output = next(v for v in variants if v[0] == label)
                model, _ = make_model(original, blocks, output)
                entry['exports'][label] = export_selected(model, inputs[-1], dest/(label+'.onnx'))
                del model

        records['scenes'][scene] = entry
        (a.out/'records.json').write_text(json.dumps(records, indent=2))
        del original, inputs
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
