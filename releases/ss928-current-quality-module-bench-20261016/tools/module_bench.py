"""Prepare delivered real tensor cases, compile modules, and summarize board results."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {'FLOAT': np.dtype('<f4'), 'FLOAT16': np.dtype('<f2')}


def catalog():
    return json.loads((ROOT/'PACKAGE_INDEX.json').read_text())


def graph_for(key):
    record = next(item for item in catalog()['graphs'] if item['key'] == key)
    folder = ROOT/record['folder']
    return record, folder, json.loads((folder/'manifest.json').read_text())


def write_case(folder, manifest, arrays):
    folder.mkdir(parents=True, exist_ok=True)
    for side in ['inputs', 'outputs']:
        lines = [str(len(manifest[side]))]
        for value in manifest[side]:
            dtype = DTYPES[value['dtype']]
            shape = value['shape']
            if any(dimension <= 0 for dimension in shape):
                raise ValueError('Static deployment tensor shape required: '+value['name'])
            rows = int(np.prod(shape[:-1])) if shape else 1
            width = int(shape[-1]) if shape else 1
            lines.append('%s %s %d %d %d' % (value['name'], value['dtype'], dtype.itemsize, rows, width))
            if side == 'inputs':
                array = np.asarray(arrays[value['name']], dtype=dtype)
                if list(array.shape) != shape or not np.isfinite(array).all():
                    raise ValueError('Input shape or finite-value mismatch: '+value['name'])
                array.tofile(folder/(value['name']+'.bin'))
        (folder/(side+'_spec.txt')).write_text('\n'.join(lines)+'\n')


def prepare(args):
    index = catalog()
    script = ['#!/bin/sh', 'set -eu', 'BENCH=${SS928_RUNNER:-./module_bench}',
        'OM_ROOT=${OM_ROOT:-./compiled}', 'INPUT_ROOT=${INPUT_ROOT:-./inputs}',
        'RESULT_ROOT=${RESULT_ROOT:-./board_results}', 'WARMUP=${WARMUP:-50}', 'RUNS=${RUNS:-100}']
    for record in index['graphs']:
        _, folder, manifest = graph_for(record['key'])
        key = record['key']
        target = args.out/key
        smoke = np.load(folder/manifest.get('smoke_inputs', 'val_graph_inputs.npz'))
        write_case(target/'smoke', manifest, smoke)
        calibration = manifest.get('calibration_files', ['train_graph_inputs.npz'])
        lists = {value['name']: [] for value in manifest['inputs']}
        for sample, filename in enumerate(calibration):
            arrays = np.load(folder/filename)
            sample_folder = target/('calibration_%02d'%sample)
            write_case(sample_folder, manifest, arrays)
            for name in lists:
                lists[name].append(str((sample_folder/(name+'.bin')).resolve()))
        for name, paths in lists.items():
            (target/(name+'.list')).write_text('\n'.join(paths)+'\n')
        (target/'calibration_lists.json').write_text(json.dumps(lists, indent=2))
        script.extend(['test -f "$OM_ROOT/%s/model.om"'%key,
            'mkdir -p "$RESULT_ROOT/%s"'%key,
            '"$BENCH" "$OM_ROOT/%s/model.om" "$INPUT_ROOT/%s/smoke" "$RESULT_ROOT/%s" "$WARMUP" "$RUNS"'%(key, key, key)])
        print('PREPARED', key)
    (args.out/'run_components.sh').write_text('\n'.join(script)+'\n')


def compile_graph(args):
    _, folder, manifest = graph_for(args.graph)
    if any(value['dtype'] not in DTYPES for value in manifest['inputs']+manifest['outputs']):
        raise ValueError('SDK interface requires FP16 or FP32; check graph export precision')
    def shape(value):
        return value['name']+':'+','.join(map(str, value['shape']))
    input_types = ';'.join(value['name']+':'+('FP16' if value['dtype'] == 'FLOAT16' else 'FP32') for value in manifest['inputs'])
    output_types = {value['dtype'] for value in manifest['outputs']}
    if len(output_types) != 1:
        raise ValueError('Set mixed output precision explicitly with the installed SDK')
    command = [args.atc, '--framework=5', '--soc_version=SS928V100', '--npu_arch=V101',
        '--model='+str((folder/'model.onnx').resolve()), '--input_shape='+';'.join(shape(value) for value in manifest['inputs']),
        '--input_type='+input_types, '--output_type='+('FP16' if output_types == {'FLOAT16'} else 'FP32'),
        '--image_list='+args.image_list, '--compile_mode='+args.compile_mode,
        '--online_model_type=3', '--log_level=1', '--output='+str(args.out.resolve()/'model')]
    print(json.dumps(command))
    if not args.print_only:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/'command.json').write_text(json.dumps(command, indent=2))
        with (args.out/'conversion.log').open('wb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def compare(args):
    _, folder, manifest = graph_for(args.graph)
    pc = np.load(folder/'pc_output.npz')
    result = []
    for index, output in enumerate(manifest['outputs']):
        dtype = DTYPES[output['dtype']]
        actual = np.fromfile(args.board_output/('output_%03d.bin'%index), dtype=dtype)
        expected = pc[output['name']]
        if actual.size != expected.size or not np.isfinite(actual).all():
            raise ValueError('Output size or finite-value mismatch')
        error = actual.astype(np.float32).reshape(expected.shape)-expected.astype(np.float32)
        result.append(dict(output=output['name'], MAE=float(np.abs(error).mean()),
            max_absolute=float(np.abs(error).max()), P95_absolute=float(np.percentile(np.abs(error), 95)),
            mean_bias=float(error.mean())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(graph=args.graph, results=result), indent=2))
    print(args.out)


def summarize(args):
    records, measured = [], {}
    profile = {}
    if args.profile_csv:
        with args.profile_csv.open() as stream:
            for row in csv.DictReader(stream):
                if row.get('phase', 'measured') == 'measured':
                    profile.setdefault(row['graph'], []).append(float(row['pure_npu_ms']))
    for graph in catalog()['graphs']:
        filename = args.results/graph['key']/'timing.csv'
        if not filename.exists():
            records.append(dict(graph=graph['key'], status='missing'))
            continue
        with filename.open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['phase'] == 'measured']
        values = [float(row['host_execute_ms']) for row in rows]
        if not values:
            raise ValueError('No measured rows: '+str(filename))
        stat = dict(graph=graph['key'], status='measured', samples=len(values), mean_ms=float(np.mean(values)), P95_ms=float(np.percentile(values, 95)))
        if graph['key'] in profile:
            samples = profile[graph['key']]
            stat['profiler_pure_NPU'] = dict(samples=len(samples), mean_ms=float(np.mean(samples)), P95_ms=float(np.percentile(samples, 95)))
        records.append(stat)
        measured[graph['key']] = stat
    workloads = []
    for path in (ROOT/'workloads').glob('*/val_frame_019.json'):
        plan = json.loads(path.read_text())
        keys = {plan['scene']+'/'+graph: count for graph, count in plan['counts'].items()}
        missing = [key for key in keys if key not in measured]
        estimate = None if missing else sum(count*measured[key]['mean_ms'] for key, count in keys.items())
        profile_missing = [key for key in keys if key not in profile]
        profile_estimate = None if profile_missing else sum(count*float(np.mean(profile[key])) for key, count in keys.items())
        workloads.append(dict(scene=plan['scene'], frame=19, calls=sum(keys.values()), missing=missing,
            sum_of_component_mean_host_execute_ms=estimate, sum_of_component_mean_pure_NPU_ms=profile_estimate,
            missing_profiler=profile_missing, full_pipeline_time_ms=None, full_frame_P95_ms=None))
    result = dict(boundary='SDK synchronous module execute; not profiler pure NPU or full pipeline',
        modules=records, workload_estimates=workloads, full_pipeline_measured=False,
        CPU_tracking_mask_and_fusion_not_measured=True,
        P95_note='Per-module P95 only; do not add module P95 values to claim a full-frame P95')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(args.out)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    item = commands.add_parser('prepare')
    item.add_argument('--out', type=Path, required=True)
    item = commands.add_parser('compile')
    item.add_argument('--graph', required=True)
    item.add_argument('--atc', required=True)
    item.add_argument('--image-list', required=True, help='Installed SDK multi-input image_list argument; prepared training lists only')
    item.add_argument('--compile-mode', default='1')
    item.add_argument('--out', type=Path, required=True)
    item.add_argument('--print-only', action='store_true')
    item = commands.add_parser('compare')
    item.add_argument('--graph', required=True)
    item.add_argument('--board-output', type=Path, required=True)
    item.add_argument('--out', type=Path, required=True)
    item = commands.add_parser('summarize')
    item.add_argument('--results', type=Path, required=True)
    item.add_argument('--profile-csv', type=Path)
    item.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'list':
        for graph in catalog()['graphs']:
            print(graph['key'], graph['module_type'])
    else:
        {'prepare': prepare, 'compile': compile_graph, 'compare': compare, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
