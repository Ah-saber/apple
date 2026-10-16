"""Validation cohorts for tiny and low-gap GT targets; no semantic labels."""
import numpy as np
import torch
from gap_metrics_v5 import GapMetrics
from target_validation_rows_v2 import select_rows, key, history_frames
from run_reduced_v2 import read_gt
from train_guided import score as base_score
from run_verified import permitted as base_permitted


def cohorts(records, names):
    result = {name: {} for name in names}
    for group, keep in [('tiny', lambda r: r['target_pixels'] <= 9),
                        ('weak', lambda r: abs(r['gt_gap_gray']) <= 15)]:
        chosen = [r for r in records if keep(r)]
        for name in names:
            values = [r['methods'][name] for r in chosen]
            result[name][group] = {'count': len(values),
                'gap_error_gray': float(np.mean([v['gap_error_gray'] for v in values])) if values else None,
                'polarity_inverted': float(np.mean([v['polarity_inverted'] for v in values])) if values else None}
    return result


def permitted(candidate, reference):
    ok = base_permitted(candidate, reference)
    for group in ['tiny', 'weak']:
        c = candidate.get('cohorts', {}).get(group, {})
        r = reference.get('cohorts', {}).get(group, {})
        if c.get('count', 0) and r.get('count', 0):
            ok = ok and c['gap_error_gray'] <= r['gap_error_gray'] + .05
            ok = ok and c['polarity_inverted'] <= r['polarity_inverted'] + 1e-6
    return bool(ok)


def score(metrics):
    value = base_score(metrics)
    for cohort in metrics.get('cohorts', {}).values():
        if cohort['count']:
            value += .5 * cohort['gap_error_gray']
    return value


def validation(models, config, data, scenes, box_for):
    result = {}
    with torch.inference_mode():
        for scene in scenes:
            rows, templates = select_rows(data, config, scene)
            assert rows
            observations = []
            for row in rows:
                metric = GapMetrics(row['eval_crop_tlhw'], models)
                metric.background, metric.noise = templates[key(row)]
                frame = int(row['frame_id'])
                for f in history_frames(data,row):
                    rr = dict(row, frame_id=f)
                    raw = data.normalized_stack(rr, (0, 0, 1024, 1280))[None].cuda()
                    context, _ = data.context_for(rr)
                    context = context[None].cuda()
                    box = box_for(rr).cuda()
                    ps = {name: (m.native(raw, context, box)[0, 0].float().clamp(0, 1)*255).cpu().numpy()
                          for name, m in models.items()}
                    metric.update(f, read_gt(config, row, f), ps)
                observations.append(metric.report())
            records = [r for report in observations for r in report['targets']]
            grouped = cohorts(records, models)
            result[scene] = {}
            for name in models:
                frames = [r['frames'][-1]['methods'][name] for r in observations]
                targets = [r['methods'][name] for r in records]
                keys = set().union(*(f.keys() for f in frames))
                result[scene][name] = {
                    'frames': {k: float(np.mean([f[k] for f in frames if k in f])) for k in keys},
                    'targets': {k: float(np.mean([r[k] for r in targets])) for k in targets[0]} if targets else {},
                    'target_observations': len(targets), 'cohorts': grouped[name]}
    return result




