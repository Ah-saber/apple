"""Audit frozen teacher profiles and S0 coverage; do not change dataset splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(workspace):
    upload_path = workspace / 'docs/plans/ss928-sr-selection/2026-09-21-enhanced-upload-manifest.json'
    profile_path = workspace / 'docs/reports/ITERATION-20260921-V6/manifest.json'
    groups_path = workspace / 'huawei_night_code/ss928_sr/manifests/split_s0/groups.jsonl'
    pairs_path = workspace / 'docs/evidence/ss928-sr-selection/TASK-018/metadata/manifests/dataset_d0/pairs.jsonl'
    uploads = {r['sequence_id']: r for r in read_json(upload_path)['entries']}
    profiles = {r['sequence']: r for r in read_json(profile_path)['sequences']}
    groups = [json.loads(line) for line in groups_path.read_text().splitlines()]
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    counts = Counter((r['domain'], r['sequence_id']) for r in pairs)
    if len({r['sample_id'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate sample identifiers')
    labels = {
        'day_normal': '正常白天',
        'weather_light': '天气轻档',
        'weather_medium': '天气中档',
        'weather_heavy': '天气重档',
        'weather_heavy_c32': '天气重档加宽尺度补偿',
        'night_ordinary': '普通夜间',
        'night_special': '特殊夜间',
    }
    records = []
    # Candidate repairs coverage of the five sufficiently represented profiles.
    # This does not release a new split or solve scarce-profile coverage.
    proposal_moves = {'adverse_capture_00': 'val', 'adverse_capture_01': 'test'}
    for group in groups:
        for sequence in group['sequences']:
            domain = group['domain']
            row = {'domain': domain, 'sequence_id': sequence,
                   'frames': counts[domain, sequence], 'capture_group': group['group_id'],
                   'split_s0': group['split'],
                   'candidate_split': proposal_moves.get(group['group_id'], group['split'])}
            if domain == 'normal':
                scene = 'day_normal'
                row['teacher_version'] = 'camera_jpeg_v1'
            else:
                source = uploads[sequence]
                result_path = (workspace / source['source_enhanced_directory']).parent / 'result.json'
                result = read_json(result_path)
                if result['sequence'] != sequence or result['frames'] != row['frames']:
                    raise ValueError('Teacher identity/frame mismatch')
                row.update(teacher_version=source['teacher_version'],
                           teacher_experiment=source['source_experiment'],
                           teacher_result_path=result_path.relative_to(workspace).as_posix(),
                           teacher_result_sha256=sha256(result_path),
                           recorded_parameters=result['parameters'])
                if domain == 'adverse':
                    profile = result['profile']
                    if profile != profiles[sequence]['profile']:
                        raise ValueError('Teacher profile differs from frozen manifest')
                    scene = {'W-light': 'weather_light', 'W-medium': 'weather_medium',
                             'W-heavy': 'weather_heavy'}[profile['profile_id']]
                    if profile['coarse_sigma']:
                        if scene != 'weather_heavy' or profile['coarse_sigma'] != 32:
                            raise ValueError('Unrecognized coarse-correction profile')
                        scene += '_c32'
                    row.update(profile=profile, grain=result['grain'],
                               effective_window_frames=2 * result['grain']['radius'] + 1)
                elif source['teacher_version'] == 'v5.1':
                    scene = 'night_ordinary'
                    row['effective_window_frames'] = 2 * result['parameters']['radius'] + 1
                elif source['teacher_version'] == 'v8':
                    scene = 'night_special'
                    row['detail_profile'] = result['profile']
                else:
                    raise ValueError('Unrecognized night teacher')
            row['scene_id'] = scene
            records.append(row)
    identities = {(r['domain'], r['sequence_id']) for r in records}
    if identities != set(counts) or len(records) != len(identities):
        raise ValueError('Missing or duplicate sequence assignment')
    assert len(records) == 75 and sum(r['frames'] for r in records) == 8669
    assert {r['scene_id'] for r in records} == set(labels)
    summaries = []
    for scene, label in labels.items():
        rows = [r for r in records if r['scene_id'] == scene]
        summary = {'scene_id': scene, 'label': label, 'sequences': len(rows),
                   'frames': sum(r['frames'] for r in rows),
                   'capture_groups': sorted({r['capture_group'] for r in rows})}
        for field in ('split_s0', 'candidate_split'):
            summary[field] = {split: {
                'sequences': sum(r[field] == split for r in rows),
                'frames': sum(r['frames'] for r in rows if r[field] == split),
                'capture_groups': sorted({r['capture_group'] for r in rows if r[field] == split})
            } for split in ('train', 'val', 'test')}
        summary['minimum_additional_groups_for_three_way_split'] = max(0, 3 - len(summary['capture_groups']))
        summaries.append(summary)
    return {'task': 'TASK-018', 'date': '2026-09-22',
            'status': 'audit_only_candidate_not_released',
            'grouping_basis': 'fixed algorithm and explicit hyperparameters, not physical weather labels',
            'same_scene_development_is_not_independent_generalization': True,
            'sources': [{'path': p.relative_to(workspace).as_posix(), 'sha256': sha256(p)}
                        for p in (upload_path, profile_path, groups_path, pairs_path)],
            'candidate_group_moves': proposal_moves,
            'candidate_limit': 'C32 and special night still lack independent three-way coverage',
            'scenes': summaries, 'sequences': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'sequences': len(result['sequences']), 'scene_profiles': len(result['scenes']),
                      'status': result['status'], 'output': str(args.output)}, ensure_ascii=False))
