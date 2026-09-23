"""Build independent, train-region-only float DN targets using frozen traditional kernels."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Timings:
    def __init__(self):
        self.rows = []

    @contextmanager
    def section(self, module):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append({'module': module, 'seconds': time.perf_counter() - start})


def load_teacher(release):
    path = release / 'runtime/huawei_weather_code/server_v8_sync/run_collection.py'
    spec = importlib.util.spec_from_file_location('midgt_frozen_collection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def region_input(raw, frames, roi):
    top, left, height, width = roi
    if top != 0 or height != 1024 or left not in (0, 256) or width not in (768, 1280):
        raise ValueError('Unreviewed sensor-region geometry')
    if frames != list(range(raw.shape[0])):
        raise ValueError('Teacher requires a complete contiguous training sequence')
    # Slice BEFORE converting/copying or calculating ANY reference/statistic.
    return np.asarray(raw[:, top:top+height, left:left+width], dtype=np.float32).copy()


def denoise_region(module, item, raw, roi, timing):
    c, gc = module.configuration(item)
    if raw.dtype != np.float32 or list(raw.shape[1:]) != roi[2:]:
        raise ValueError('Teacher must receive only its allowed region')
    output = np.empty_like(raw)
    if item['branch'] == 'special-night-v8':
        with timing.section('dynamic_bias_prepare'):
            prepared, _ = module.prepare_video(raw, c)
        with timing.section('trajectory_fusion'):
            for k, (value, _) in enumerate(module.fuse(prepared, c, gc)):
                output[k] = value
        with timing.section('structure_initialization'):
            stage = module.TextureV8(output.mean(0), 'natural')
            assert stage.parameters() == item['texture']
            stage.c = dataclasses.replace(stage.c, detail=False)
        with timing.section('structure_denoise_without_detail'):
            for k in range(len(output)):
                output[k] = stage.process(output[k])
        return output, {'segments': [[0, len(raw)]], 'edge': 'not applicable', 'detail': False}
    with timing.section('segmentation'):
        splits = module.segments(raw)
    logs = []
    for begin, end in splits:
        source = raw[begin:end]
        if item['branch'] == 'weather-v6':
            with timing.section('fixed_left_bias'):
                if roi[1] == 0:
                    bias, edge = module.estimate_edge(source.mean(0), **item['profile']['edge'])
                else:
                    # The calibrated sensor-left fitting area is outside this ROI.
                    # Do not fit the crop edge as though it were the sensor edge.
                    bias = np.zeros(raw.shape[1:], np.float32)
                    edge = {'active': False, 'reason': 'sensor-left calibration area excluded by training split'}
            if item['profile']['coarse_sigma']:
                with timing.section('C32_inside_training_region'):
                    source, _ = module.correct_broad_field(source, sigma=item['profile']['coarse_sigma'])
            with timing.section('dynamic_bias_prepare'):
                prepared, _ = module.prepare_video(source, c)
                prepared -= bias[None]
            with timing.section('trajectory_fusion'):
                for k, (value, _) in enumerate(module.fuse(prepared, c, gc), begin):
                    output[k] = value
            logs.append(edge)
        else:
            # Same prepare + adaptive kernel called by frozen denoise_video.
            import video_v5
            with timing.section('dynamic_bias_prepare'):
                prepared, _ = module.prepare_video(source, c)
            assert c.fusion == 'adaptive'
            with timing.section('ordinary_adaptive_fusion'):
                for k, (value, _) in enumerate(video_v5.fuse_adaptive(prepared, c), begin):
                    output[k] = value
    return output, {'segments': splits, 'edge': logs, 'detail': False}


def regression(module, items, data, preview, destination):
    rows = []
    original = json.loads((preview / 'run.json').read_text())
    for record in original['results']:
        start = time.perf_counter()
        item = items[record['sequence']]
        sequence = module.make_sequence(item, data)
        raw = np.stack([sequence.raw(k) for k in range(sequence.frames)])
        got, _ = denoise_region(module, item, raw, [0, 0, 1024, 1280], Timings())
        expected = np.load(record['middle_gt'], mmap_mode='r')
        delta = np.abs(got - expected)
        row = {'sequence': item['sequence'], 'max_abs_dn': float(delta.max()),
               'array_equal': bool(np.array_equal(got, expected)), 'seconds': time.perf_counter()-start}
        assert row['array_equal'], row
        rows.append(row)
        print(json.dumps({'event': 'preview_regression_passed', **row}), flush=True)
    save(destination / 'preview_regression.json', {'status': 'passed', 'rows': rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preview', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2); cv2.ocl.setUseOpenCL(False)
    start = time.perf_counter()
    sources = json.loads((args.release / 'manifests/source-files.json').read_text())['files']
    for e in sources:
        assert sha(args.release / 'runtime' / e['path']) == e['sha256']
    module = load_teacher(args.release)
    items = {r['sequence']: r for r in json.loads((args.release / 'manifests/collection.json').read_text())['sequences']}
    ids = set((args.data / 'manifests/split_s1/train.txt').read_text().splitlines())
    groups = defaultdict(list)
    for line in (args.data / 'manifests/dataset_d1/pairs.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['sample_id'] in ids and row['scene_id'] != 'day_normal':
            groups[row['domain'] + '/' + row['sequence_id']].append(row)
    report = {'status': 'building', 'dataset': 'dataset_d1', 'version': 'middle_raw_train_region_v1',
              'source_release': str(args.release), 'source_files': sources, 'script_sha256': sha(__file__),
              'source_collection_sha256': sha(args.release / 'manifests/collection.json'),
              'train_split_sha256': sha(args.data / 'manifests/split_s1/train.txt'),
              'pairs_sha256': sha(args.data / 'manifests/dataset_d1/pairs.jsonl'),
              'numpy': np.__version__, 'opencv': cv2.__version__, 'python': sys.version,
              'units': 'float32 DN, no clipping or display mapping',
              'scope': 'complete train sequences; all statistics computed strictly inside train ROI; no val/test targets',
              'region_adaptation': 'central ROI omits sensor-left fitted correction; C32 and all statistics recomputed inside ROI',
              'sequences': []}
    save(args.output / 'index.json', report)
    try:
        regression(module, items, args.data, args.preview, args.output)
        for number, (key, rows) in enumerate(sorted(groups.items()), 1):
            begin = time.perf_counter(); timing = Timings()
            rows.sort(key=lambda r: r['frame_id'])
            first = rows[0]; item = items[first['sequence_id']]
            roi = first['train_roi_tlhw']; frames = [r['frame_id'] for r in rows]
            assert all(r['train_roi_tlhw'] == roi and r['raw']['path'] == first['raw']['path'] for r in rows)
            with timing.section('source_identity_and_region_copy'):
                assert sha(args.data / item['server_raw']) == item['raw_sha256']
                decoded_path = args.data / first['raw']['path']
                decoded_sha = sha(decoded_path)
                decoded = np.load(decoded_path, mmap_mode='r', allow_pickle=False)
                seq = module.make_sequence(item, args.data)
                assert len(decoded) == seq.frames
                for k in frames:
                    assert np.array_equal(decoded[k], seq.raw(k)), (key, k)
                raw = region_input(decoded, frames, roi)
            print(json.dumps({'event': 'start_sequence', 'number': number, 'total': len(groups), 'key': key, 'roi': roi}), flush=True)
            middle, meta = denoise_region(module, item, raw, roi, timing)
            assert middle.shape == raw.shape and middle.dtype == np.float32 and np.isfinite(middle).all()
            with timing.section('write_and_verify'):
                dest = args.output / key / 'middle_dn.npy'; dest.parent.mkdir(parents=True)
                np.save(dest, middle, allow_pickle=False)
                assert np.array_equal(np.load(dest, mmap_mode='r'), middle)
                digest = sha(dest)
            entry = {'key': key, 'scene_id': first['scene_id'], 'path': str(dest.relative_to(args.output)),
                     'sha256': digest, 'shape': list(middle.shape), 'roi_tlhw': roi, 'frame_ids': frames,
                     'sample_ids': [r['sample_id'] for r in rows], 'source_raw_path': first['raw']['path'],
                     'source_decoded_sha256': decoded_sha, 'source_raw_sha256': item['raw_sha256'],
                     'configuration': {k:v for k,v in item.items() if k not in ('baseline_files',)},
                     'effective_processing': meta, 'timing_modules': timing.rows, 'wall_seconds': time.perf_counter()-begin}
            report['sequences'].append(entry)
            save(args.output / 'index.json', report)
            print(json.dumps({'event': 'sequence_complete', 'key': key, 'seconds': entry['wall_seconds']}), flush=True)
            del raw, middle, decoded
        for e in sources:
            assert sha(args.release / 'runtime' / e['path']) == e['sha256']
        report.update(status='complete', source_after_verified=True,
                      total_frames=sum(len(r['frame_ids']) for r in report['sequences']),
                      wall_seconds=time.perf_counter()-start)
        assert report['total_frames'] == 2820
    except BaseException as error:
        report.update(status='failed', error=repr(error), wall_seconds=time.perf_counter()-start)
        raise
    finally:
        save(args.output / 'index.json', report)


if __name__ == '__main__':
    main()
