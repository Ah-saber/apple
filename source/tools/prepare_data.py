"""Build immutable CPU dataset D0 from verified source catalog and teacher archive."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw

H, W = 1024, 1280
VERSION = 'raw-display-d0-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_new(path, obj, lines=False):
    path = Path(path)
    data = (''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in obj) if lines
            else json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    put(path, data)


def put(path, data):
    path = Path(path)
    if path.exists():
        if sha(path) != hashlib.sha256(data).hexdigest():
            raise RuntimeError('Refusing to replace different file: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temporary.rename(path)


def safe_child(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe relative path')
    q = (root / relative).resolve()
    if not q.is_relative_to(root.resolve()):
        raise ValueError('Path escapes data root')
    return q


def capture_seconds(s):
    name = s['sequence_id']
    if s['domain'] == 'night':
        m = re.fullmatch(r'raw_(\d{8})_(\d{2})(\d{2})(\d{2})', name)
        if not m:
            return None
        return int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])
    pattern = r'04-25-(\d{2})-(\d{2})-(\d{2})' if s['domain'] == 'normal' else r'_1-(\d{2})-(\d{2})-(\d{2})'
    m = re.search(pattern, name)
    if not m:
        raise ValueError('Unrecognized capture naming: ' + name)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])


def make_groups(catalog):
    groups = []
    for domain in ('normal', 'adverse', 'night'):
        ordered = sorted([s for s in catalog if s['domain'] == domain],
                         key=lambda s: (capture_seconds(s) is None, capture_seconds(s) or 0))
        previous = None
        block = None
        for s in ordered:
            sec = capture_seconds(s)
            special = sec is None
            if block is None or special or previous is None or sec - previous > 600:
                block = {'group_id': f'{domain}_capture_{sum(g["domain"] == domain for g in groups):02d}',
                         'domain': domain, 'sequences': [], 'frames': 0,
                         'special_single_sequence': special,
                         'group_basis': 'filename ordering; <=10min gap grouped; timestamps not independently certified',
                         'view_scope': 'related fixed view; no independent-scene claim'}
                groups.append(block)
            block['sequences'].append(s['sequence_id'])
            block['frames'] += s['frame_count']
            previous = sec
    for domain in ('normal', 'adverse', 'night'):
        candidates = sorted([g for g in groups if g['domain'] == domain and not g['special_single_sequence']],
                            key=lambda g: (g['frames'], g['group_id']))
        if len(candidates) < 3:
            raise RuntimeError('Need >=3 capture groups; explicit split decision required')
        for g in groups:
            if g['domain'] == domain:
                g['split'] = 'train'
        # Whole small capture groups provide within-view validation/test;
        # keep at least one ordinary training group and the special sequence.
        candidates[0]['split'] = 'val'
        candidates[1]['split'] = 'test'
        if len(candidates) > 5:
            target = sum(g['frames'] for g in candidates) * 0.10
            rest = candidates[2:-1]
            for split in ('val', 'test'):
                while sum(g['frames'] for g in candidates if g['split'] == split) < target and rest:
                    rest.pop(0)['split'] = split
    return groups


def alignment_probe(raw, gt):
    def edges(a):
        a = a.astype(np.float32).reshape(128, 8, 160, 8).mean(axis=(1, 3))
        dy, dx = np.gradient(a)
        return np.sqrt(dx * dx + dy * dy)
    r, g = edges(raw), edges(gt)
    def corr(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        a -= a.mean(); b -= b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
    scores = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            scores.append((corr(r[3:-3, 3:-3], g[3 + dy:125 + dy, 3 + dx:157 + dx]), dy, dx))
    best = max(scores)
    center = corr(r[3:-3, 3:-3], g[3:-3, 3:-3])
    return {'edge_ncc_zero': center, 'best_offset_in_8px_cells': [best[1], best[2]],
            'best_edge_ncc': best[0],
            'flip_lr_ncc': corr(r[3:-3, 3:-3], np.fliplr(g)[3:-3, 3:-3]),
            'flip_ud_ncc': corr(r[3:-3, 3:-3], np.flipud(g)[3:-3, 3:-3]),
            'scope': 'coarse spatial diagnostic only; not hardware timestamp certification'}


def extract_teachers(root, package, manifest):
    archive = package / manifest['archive']
    if archive.stat().st_size != manifest['archive_bytes'] or sha(archive) != manifest['archive_sha256']:
        raise RuntimeError('Archive SHA256/size mismatch')
    expected = {e['archive_path']: e for e in manifest['files']}
    seen = set()
    incoming = root / 'incoming/v8_snapshot_20260921'
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise RuntimeError('Unexpected archive member')
            e = expected[member.name]
            data = tar.extractfile(member).read()
            if len(data) != e['bytes'] or hashlib.sha256(data).hexdigest() != e['sha256']:
                raise RuntimeError('Teacher payload mismatch')
            put(safe_child(incoming, member.name), data)
            seen.add(member.name)
    if seen != set(expected):
        raise RuntimeError('Missing teacher archive members')
    print(json.dumps({'event': 'teachers_verified', 'files': len(seen)}), flush=True)


def build_sequence(root, s, teacher_files):
    domain, sid, n = s['domain'], s['sequence_id'], s['frame_count']
    source = safe_child(root, s['path'])
    outdir = safe_child(root, f'decoded/raw_v1/{domain}/{sid}')
    marker = outdir / 'metadata.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['source_sha256'] != s['source_sha256'] or sha(source) != s['source_sha256']:
            raise RuntimeError('Resume source changed')
        if sha(root / record['raw_path']) != record['raw_sha256']:
            raise RuntimeError('Resume RAW cache changed')
        for t in record['targets']:
            if sha(root / t['path']) != t['sha256']:
                raise RuntimeError('Resume GT changed')
        return record
    stat = source.stat()
    if stat.st_size != s['bytes'] or stat.st_size != n * s['frame_bytes']:
        raise RuntimeError('Source frame size mismatch')
    outdir.mkdir(parents=True, exist_ok=True)
    temp = outdir / 'raw_u16.partial.npy'
    if temp.exists():
        raise RuntimeError('Incomplete RAW cache exists; inspect before recovery: ' + str(temp))
    cache = np.lib.format.open_memmap(temp, mode='w+', dtype='<u2', shape=(n, H, W))
    hasher = hashlib.sha256()
    hist = np.zeros(65536, dtype=np.int64)
    targets, probes = [], []
    frames_to_probe = {0, n // 2, n - 1}
    thumbnail = None
    for t in range(n):
        if t == 0:
            stream = source.open('rb')
        data = stream.read(s['frame_bytes'])
        if len(data) != s['frame_bytes']:
            raise RuntimeError('Short RAW read')
        hasher.update(data)
        stored = np.frombuffer(data, dtype='<u2').reshape(s['stored_shape'])
        raw = stored[:, W:] if domain == 'night' else stored
        if domain == 'night':
            y = stored[:, :W].astype(np.int32) - 32768
            if y.min() < 0 or y.max() > 255:
                raise RuntimeError('Night camera Y range mismatch')
        cache[t] = raw
        hist += np.bincount(raw.ravel(), minlength=65536)
        if domain == 'normal':
            source_gt = root / s['jpeg']['directory'] / f'{sid}_{t}.jpg'
            data_gt = source_gt.read_bytes()
            with Image.open(io.BytesIO(data_gt)) as im:
                im.load()
                if im.mode != 'L' or im.size != (W, H):
                    raise RuntimeError('Normal JPEG mode/shape mismatch')
                gt = np.array(im)
                b = io.BytesIO(); im.save(b, format='PNG', compress_level=1)
                target_bytes = b.getvalue()
            target_rel = f'targets/camera_jpeg_v1/normal/{sid}/{t:06d}.png'
            teacher, exp = 'camera_jpeg_v1', None
            source_hash = hashlib.sha256(data_gt).hexdigest()
        else:
            e = teacher_files[(domain, sid, t)]
            source_gt = root / 'incoming/v8_snapshot_20260921' / e['archive_path']
            target_bytes = source_gt.read_bytes()
            if hashlib.sha256(target_bytes).hexdigest() != e['sha256']:
                raise RuntimeError('Incoming GT changed')
            with Image.open(io.BytesIO(target_bytes)) as im:
                im.load()
                if im.mode != 'L' or im.size != (W, H):
                    raise RuntimeError('Enhanced PNG mode/shape mismatch')
                gt = np.array(im)
            target_rel = f'targets/v8_snapshot_20260921/{domain}/{sid}/{t:06d}.png'
            teacher, exp = e['teacher_version'], e['source_experiment']
            source_hash = e['sha256']
        target = safe_child(root, target_rel)
        put(target, target_bytes)
        with Image.open(target) as verified:
            if not np.array_equal(np.asarray(verified), gt):
                raise RuntimeError('Target pixels changed during import')
        targets.append({'path': target_rel, 'sha256': hashlib.sha256(target_bytes).hexdigest(),
                        'source_path': source_gt.relative_to(root).as_posix(), 'source_sha256': source_hash,
                        'teacher_version': teacher, 'source_experiment': exp, 'frame_id': t})
        if t in frames_to_probe:
            probes.append({'frame_id': t, **alignment_probe(raw, gt)})
        if t == 0:
            lo, hi = np.percentile(raw[::4, ::4], [1, 99])
            visible = np.uint8(np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255))
            thumbnail = Image.new('L', (320, 128))
            thumbnail.paste(Image.fromarray(visible).resize((160, 128)), (0, 0))
            thumbnail.paste(Image.fromarray(gt).resize((160, 128)), (160, 0))
    if stream.read(1):
        raise RuntimeError('Unexpected trailing bytes')
    stream.close()
    if hasher.hexdigest() != s['source_sha256']:
        raise RuntimeError('Full RAW SHA256 differs from original project: ' + sid)
    after = source.stat()
    if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError('Source changed during read')
    cache.flush(); del cache
    final = outdir / 'raw_u16.npy'
    if final.exists():
        raise RuntimeError('RAW cache already exists without completion record')
    temp.rename(final)
    histogram_path = outdir / 'dn_histogram.json'
    nonzero = np.flatnonzero(hist)
    histogram = [[int(i), int(hist[i])] for i in nonzero]
    dump_new(histogram_path, histogram)
    record = {'domain': domain, 'sequence_id': sid, 'frame_count': n,
              'source_path': s['path'], 'source_sha256': hasher.hexdigest(),
              'format_id': s['format_id'], 'raw_path': final.relative_to(root).as_posix(),
              'raw_sha256': sha(final), 'raw_shape': [n, H, W], 'raw_dtype': '<u2',
              'raw_min': int(nonzero.min()), 'raw_max': int(nonzero.max()),
              'histogram_path': histogram_path.relative_to(root).as_posix(),
              'targets': targets, 'alignment_probes': probes}
    thumb = io.BytesIO(); thumbnail.save(thumb, format='JPEG', quality=80)
    put(root / f'qa/dataset_d0/previews/{domain}/{sid}.jpg', thumb.getvalue())
    dump_new(marker, record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--run-id', default='EXP-TASK-018-001')
    args = ap.parse_args()
    root, package = args.root.resolve(), args.package.resolve()
    start = time.time()
    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError('Insufficient free space')
    catalog = json.loads((package / 'source_catalog.json').read_text())
    upload = json.loads((package / 'upload_manifest.json').read_text())
    qa = root / 'qa/dataset_d0'
    identity = {'version': VERSION, 'run_id': args.run_id,
                'source_catalog_sha256': sha(package / 'source_catalog.json'),
                'upload_manifest_sha256': sha(package / 'upload_manifest.json'),
                'processor_sha256': sha(Path(__file__))}
    marker = qa / 'build_identity.json'
    if not marker.exists():
        for p in ('decoded/raw_v1', 'targets/camera_jpeg_v1', 'targets/v8_snapshot_20260921',
                  'manifests/source_v1', 'manifests/dataset_d0', 'manifests/split_s0'):
            if (root / p).exists():
                raise RuntimeError('Refusing unowned existing dataset version: ' + p)
    dump_new(marker, identity)
    extract_teachers(root, package, upload)
    teacher_files = {(e['domain'], e['sequence_id'], e['frame_id']): e for e in upload['files']}
    if len(teacher_files) != 4260 or len(catalog) != 75:
        raise RuntimeError('Catalog cardinality mismatch')
    groups = make_groups(catalog)
    assignments = {(g['domain'], sid): g for g in groups for sid in g['sequences']}
    records = []
    for s in catalog:
        rec = build_sequence(root, s, teacher_files)
        records.append(rec)
        print(json.dumps({'event': 'sequence_complete', 'index': len(records), 'domain': rec['domain'],
                          'sequence': rec['sequence_id'], 'frames': rec['frame_count'],
                          'elapsed_s': round(time.time() - start, 1)}), flush=True)
    pairs, lists, targets = [], defaultdict(list), defaultdict(list)
    training_hist = np.zeros(65536, dtype=np.int64)
    for r in records:
        group = assignments[r['domain'], r['sequence_id']]
        if group['split'] == 'train':
            for value, count in json.loads((root / r['histogram_path']).read_text()):
                training_hist[value] += count
        for t in r['targets']:
            sample_id = f'{r["domain"]}/{r["sequence_id"]}/{t["frame_id"]:06d}'
            pair = {'sample_id': sample_id, 'domain': r['domain'], 'sequence_id': r['sequence_id'],
                    'frame_id': t['frame_id'], 'group_id': group['group_id'],
                    'raw': {'path': r['raw_path'], 'index': t['frame_id'], 'format_id': r['format_id']},
                    'target': t, 'pair_status': 'source_identity_and_index_verified',
                    'alignment_limit': 'coarse spatial checks; capture hardware synchronization not certified'}
            pairs.append(pair)
            lists[group['split']].append(sample_id)
            if group['special_single_sequence']:
                lists['diagnostic'].append(sample_id)
            targets['camera_jpeg_v1' if r['domain'] == 'normal' else 'v8_snapshot_20260921'].append({'sample_id': sample_id, **t})
    x = np.arange(65536, dtype=np.float64)
    count = int(training_hist.sum())
    mean = float(np.dot(x, training_hist) / count)
    std = float(np.sqrt(np.dot((x - mean)**2, training_hist) / count))
    recipe = {'version': 'preprocess_v1', 'scale': 3, 'train_crop_hr': 384,
              'validation_crop_tlhw': [2, 0, 1020, 1278],
              'raw_normalization': {'kind': 'fixed_affine', 'offset': mean, 'scale': max(std, 1.0),
                                    'estimated_from': 'train split only; full original RAW pixels',
                                    'fit_pixels': count, 'clipping': False},
              'raw_downsample': 'nonoverlapping 3x3 area mean in float32 DN before normalization',
              'target_normalization': 'float32 / 255', 'geometric_augmentation': 'aligned random crop only',
              'seed': 928, 'domain_sampling': 'equal domain then equal sequence then uniform frame',
              'data_loader': {'persistent_workers': False, 'set_epoch_before_new_workers': True},
              'inference_raw_hw': [1024, 1280], 'inference_output_hw': [3072, 3840]}
    dump_new(root / 'recipes/preprocess_v1.json', recipe)
    dump_new(root / 'manifests/source_v1/sequences.jsonl', catalog, lines=True)
    dump_new(root / 'manifests/raw_v1/decoded.jsonl', [{k: v for k, v in r.items() if k != 'targets'} for r in records], lines=True)
    for version, rows in targets.items():
        dump_new(root / f'manifests/{version}/targets.jsonl', rows, lines=True)
    dump_new(root / 'manifests/dataset_d0/pairs.jsonl', pairs, lines=True)
    dump_new(root / 'manifests/dataset_d0/excluded.jsonl', [], lines=True)
    dump_new(root / 'manifests/split_s0/groups.jsonl', groups, lines=True)
    for split in ('train', 'val', 'test', 'diagnostic'):
        put(root / f'manifests/split_s0/{split}.txt', ('\n'.join(lists[split]) + '\n').encode())
    protocol = {'kind': 'within_fixed_view_capture_group_holdout',
                'grouping': 'filename-time gaps <=10min grouped; not independently certified timestamps',
                'assignment': 'small complete groups reserved, >=1 ordinary training group per domain',
                'scene_generalization': 'not evaluated; related fixed view shared across splits',
                'teacher_design_holdout': False,
                'teacher_limit': 'traditional teachers were developed before this split; not independent teacher evaluation',
                'diagnostic': 'special night single sequence included in train; diagnostic is not held-out',
                'split_counts': {k: len(v) for k, v in lists.items()}}
    dump_new(root / 'manifests/split_s0/protocol.json', protocol)
    probe_flags = [{'domain': r['domain'], 'sequence_id': r['sequence_id'], **p}
                   for r in records for p in r['alignment_probes']
                   if p['best_offset_in_8px_cells'] != [0, 0] or p['edge_ncc_zero'] < 0.30]
    split_sets = {s: set(ids) for s, ids in lists.items()}
    summary = {'run_id': args.run_id, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': round(time.time() - start, 2), 'sequences': len(records),
               'frames': len(pairs), 'domain_frames': dict(Counter(r['domain'] for r in pairs)),
               'split_counts': protocol['split_counts'],
               'split_domain_frames': {s: dict(Counter(p['domain'] for p in pairs if p['sample_id'] in ids)) for s, ids in split_sets.items()},
               'upload_frames': len(teacher_files), 'raw_hashes_verified': len(records),
               'targets_full_decoded_and_pixel_verified': len(pairs), 'alignment_probes': 225,
               'alignment_review_flags': probe_flags,
               'raw_normalization': recipe['raw_normalization'],
               'gpu_used': False, 'status': 'data_built_pending_independent_verification'}
    dump_new(qa / 'build_summary.json', summary)
    important = list((root / 'manifests').rglob('*')) + list((root / 'recipes').rglob('*'))
    dump_new(qa / 'manifest_sha256.json', [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in sorted(important) if p.is_file()])
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
