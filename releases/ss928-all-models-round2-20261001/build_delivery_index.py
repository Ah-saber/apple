"""Create a repository-relative index from frozen, verified experiment metadata."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    dest = repo / 'releases/ss928-all-models-round2-20261001'
    old = repo / 'releases/ss928-board-v13-20260930'
    report = repo / 'reports/ss928_all_models_round2_20260930'
    original = json.loads((old / 'MODEL_INDEX.json').read_text())['models']
    original = {row['path']: row for row in original}
    five = json.loads((report / 'results/exact_layout_five_r2/all_sequence_verification.json').read_text())['scenes']
    night = json.loads((report / 'results/exact_layout_night_r2/all_sequence_verification.json').read_text())['scenes']
    motion = json.loads((report / 'results/candidate_float_exports/candidate_float_exports.json').read_text())['candidates']
    models = []
    scenes = ['day_normal', 'weather_light', 'weather_medium', 'weather_heavy',
              'weather_heavy_c32', 'night_ordinary', 'night_special']
    for scene in scenes:
        is_night = scene.startswith('night_')
        filename = {'night_ordinary': 'ordinary_reference_after12_rows32.onnx',
                    'night_special': 'special_lowref_k5_rows32.onnx'}.get(scene)
        old_rel = 'onnx/night_preserved/' + filename if is_night else f'onnx/equivalent/{scene}/{scene}_equivalent_fp16_rows32.onnx'
        control = original[old_rel]
        folder = 'night_calibration_final' if is_night else 'deployment_corrected'
        entries = [('original_float', old / old_rel, control, control['sha256'])]
        row = night[scene] if is_night else five[scene + '_half']
        layout = row['graphs']['float_repeat']
        layout_path = report / 'results' / ('exact_layout_night_r2' if is_night else 'exact_layout_five_r2') / Path(layout['path']).name
        entries.append(('layout_float', layout_path, layout, layout['sha256']))
        candidate = motion[scene + '_motion']
        entries.append(('motion_float', report / 'results/candidate_float_exports' / Path(candidate['graph']['path']).name,
                        candidate['graph'], candidate['graph']['sha256']))
        for role, path, meta, expected in entries:
            assert sha(path) == expected, path
            def declarations(key):
                # Night exports retained the original interface; source reports explicitly record this.
                values = meta.get(key, control[key])
                type_names = {1: 'FLOAT', 2: 'UINT8', 10: 'FLOAT16'}
                return [{'name': v['name'], 'dtype': v.get('dtype', type_names.get(v.get('type'))),
                         'shape': v['shape']} for v in values]
            inputs, outputs = declarations('inputs'), declarations('outputs')
            assert outputs[0]['shape'] == [1, 1, 3072, 3840]
            assert outputs[0]['dtype'] == 'FLOAT16'
            assert inputs[0]['shape'] == [1, 9, 1024, 1280]
            models.append({'scene': scene, 'role': role, 'path': path.relative_to(repo).as_posix(),
                           'sha256': expected, 'bytes': path.stat().st_size,
                           'inputs': inputs, 'outputs': outputs,
                           'operators': meta.get('operators', []),
                           'calibration_directory_in_existing_data_bundle': f'{folder}/{scene}',
                           'interface_record': 'original declaration retained' if is_night and role != 'original_float' else 'export metadata',
                           'NPU_measured': False, 'weights_changed': role == 'motion_float' and not is_night,
                           'input_future_frames_or_GT': False})
    index = {'authorization_date': '2026-10-01', 'user_authorized_git_push': True,
             'branch': 'codex/ss928-night-nine-factor24-20260926',
             'previous_delivery_commit': '44dd59587ac6cca46923b37f86459316119932d2',
             'path_base': 'repository_root', 'NPU_measured': False,
             'target_mean_ms': 16.7, 'target_p95_ms': 16.7, 'models': models}
    (dest / 'MODEL_INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    for name in ['REVIEW.md', 'RESULTS.md', 'BOARD-VALIDATION.md']:
        content = (report / name).read_text(encoding='utf-8')
        content = content.replace('D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/',
                                  '../../reports/ss928_all_models_round2_20260930/')
        content = re.sub(r'\]\((results/[^)]+)\)', r'](../../reports/ss928_all_models_round2_20260930/\1)', content)
        content = '> 审查记录副本：文中的未推送状态对应实验核验时点；2026-10-01 用户已明确授权推送，原始审查文件继续保留。\n\n' + content
        (dest / name).write_text(content, encoding='utf-8')
    print(json.dumps({'indexed_full_float_graphs': len(models), 'scenes': len(scenes)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
