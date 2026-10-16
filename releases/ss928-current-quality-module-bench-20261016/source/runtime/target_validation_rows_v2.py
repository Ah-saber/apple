"""Choose a fixed eight-row validation set from validation GT alone."""
import cv2
import numpy as np
from gap_metrics_v5 import GapMetrics
from run_reduced_v2 import read_gt


def key(row):
    return (row['domain'], row['sequence_id'], tuple(row['eval_crop_tlhw']))


def history_frames(data,row):
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization
    name=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    frame=int(row['frame_id'])
    begin=next(s['begin'] for s in norm.by_key[name]['segments'] if s['begin']<=frame<s['end'])
    return range(max(begin,frame-5),frame+1)

def select_rows(data, config, scene):
    cache = getattr(data, '_gt_validation_cache', None)
    if cache is None:
        cache = {}
        data._gt_validation_cache = cache
    if scene in cache:
        return cache[scene]
    all_rows = [r for r in data.records if r['scene_id'] == scene]
    groups = {}
    for row in all_rows:
        groups.setdefault(key(row), {})[int(row['frame_id'])] = row
    tiny, weak, templates = [], [], {}
    for group, by_frame in groups.items():
        ordered = [by_frame[f] for f in sorted(by_frame)]
        images = [read_gt(config, row, int(row['frame_id'])) for row in ordered]
        background = GapMetrics(ordered[0]['eval_crop_tlhw'], ['GT'])
        background.prepare_background(images)
        templates[group] = (background.background, background.noise)
        # Validation records are sparse. Reconstruct each existing six-frame
        # evaluation window; do not use gaps of five frames as consecutive motion.
        for row in ordered:
            detector = GapMetrics(row['eval_crop_tlhw'], ['GT'])
            detector.background, detector.noise = templates[group]
            for f in history_frames(data, row):
                image = read_gt(config,row,f)
                detector.update(f,image,{'GT':image})
            targets = detector.report()['targets']
            if any(r['target_pixels'] <= 9 for r in targets):
                tiny.append(row)
            if any(abs(r['gt_gap_gray']) <= 15 for r in targets):
                weak.append(row)
    chosen, used = [], set()

    def add(row):
        identity = (key(row), int(row['frame_id']))
        if identity not in used:
            chosen.append(row)
            used.add(identity)

    for pool in [tiny, weak]:
        if pool:
            for i in np.unique(np.linspace(0, len(pool)-1, min(2, len(pool)), dtype=int)):
                add(pool[i])
    for i in np.unique(np.linspace(0, len(all_rows)-1, min(8, len(all_rows)), dtype=int)):
        if len(chosen) < 8:
            add(all_rows[i])
    for row in all_rows:
        if len(chosen) == min(8, len(all_rows)):
            break
        add(row)
    cache[scene] = chosen, templates
    print('GT_VALIDATION_WINDOWS', scene, [r['sample_id'] for r in chosen],
          'tiny_pool', len(tiny), 'weak_pool', len(weak), flush=True)
    return cache[scene]



