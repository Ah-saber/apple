"""Original-path causal multi-frame ablation. Keep all existing weights frozen."""
import sys, os, json, gc, argparse, time
from pathlib import Path
import numpy as np
import cv2
import torch

PROJECT = Path('/data/zhangbenzhuang/huawei_sr')
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))
from target_residual import ROOT, current_setup
from causal_detail import GuidanceView
from tieddecode_model11 import boards
from input_anchor12 import apply
from run_reduced_v2 import read_gt

DEST = Path('/tmp/rawir-motion14-20261013')
REPORT = ROOT / 'MOTION14-20261013'

def masks(nine):
    # Nine past/current RAW frames are used only to detect temporal changes.
    # The frozen SR still receives its original three/five frames.
    a = nine.numpy().copy().astype(np.float32) * 255
    current = cv2.GaussianBlur(a[-1], (0, 0), .65)
    history = []
    for raw in a[:-1]:
        old = cv2.GaussianBlur(raw, (0, 0), .65)
        old += np.median(current - old)
        history.append(old)
    history = np.stack(history)
    delta = current - history[-1]
    noise = max(.20, float(1.4826 * np.median(np.abs(delta - np.median(delta)))))
    recent = np.max(np.abs(current[None] - history[-4:]), axis=0)
    # Arrival and departure both reset temporal smoothing.
    activity = np.clip((recent / noise - 1.5) / 2, 0, 1)
    activity = cv2.dilate(activity, np.ones((5, 5), np.uint8))
    background = np.median(history, axis=0)
    residual = current - background
    local = current - cv2.GaussianBlur(current, (0, 0), 3)
    occupancy = np.clip((np.abs(residual) / noise - 1.5) / 2, 0, 1)
    occupancy *= np.clip((np.abs(local) / noise - 1) / 2, 0, 1)
    occupancy *= (residual * local > 0)
    occupancy = cv2.GaussianBlur(occupancy, (0, 0), .45)
    return activity.astype(np.float32), occupancy.astype(np.float32), noise

def run(scene, split, limit, full_sequence=False):
    REPORT.mkdir(exist_ok=True)
    cache_split=split+'_continuous' if full_sequence else split
    target = DEST / scene / cache_split
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'complete.json').exists():
        print('MOTION14_CACHE_EXISTS', scene, split, flush=True)
        return
    kind, config, df, box, count, base, original, profile = current_setup(scene)
    source = df(config, split)
    policy = apply(source, scene)
    rows = [r for r in source.records if r['scene_id'] == scene]
    groups = {}
    for r in rows:
        groups.setdefault((r['domain'], r['sequence_id']), {})[int(r['frame_id'])] = r
    summary = []
    for capture, (identity, by_frame) in enumerate(sorted(groups.items())):
        if full_sequence:
            from ir_sr.sequence_normalization import regional_key
            row=next(iter(by_frame.values()))
            norm=source.base.sequence_normalization
            key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
            contract=norm.by_key[key]
            by_frame={f:dict(row,frame_id=f) for segment in contract['segments'] for f in range(segment['begin'],segment['end'])}
        frames = sorted(by_frame)
        if limit:
            frames = frames[:limit]
        first = by_frame[frames[0]]
        out = target / ('capture_%02d' % capture)
        out.mkdir(exist_ok=True)
        if (out / 'complete.json').exists():
            summary.append(json.loads((out / 'complete.json').read_text()))
            continue
        GuidanceView.sensor = False
        data = GuidanceView(source, list(range(9-count, 9)))
        parent, model = boards(scene, base, count, kind, box(first), 500)
        del parent
        n = model.source_count
        names = ['baseline', 'current_only', 'gated_history', 'past_background', 'activity', 'occupancy', 'GT']
        arrays = {name: np.lib.format.open_memmap(out / (name+'.npy'), mode='w+', dtype=np.float16,
                  shape=(len(frames), 1024, 1280)) for name in names}
        metadata = dict(scene=scene, split=split, capture=capture, identity=identity,
                        frames=frames, row=first, input_count=n, profile=profile,
                        normalization_policy=policy, original_source_path=True,
                        context_cache_used=False, weights='TIEDDECODE500 frozen',
                        future_RAW_used=False, parameter_fit_split='train', noise=[])
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        started = time.time()
        with torch.inference_mode():
            for index, frame in enumerate(frames):
                row = by_frame[frame]
                data.current_record, data.current_crop, data.cache = row, (0,0,1024,1280), {}
                inputs = data.inputs(profile)[None].cuda()[:, -n:].half()
                context, _ = data.context_for(row)
                context = context[None].cuda().half()
                nine = data._nine_without_field().clone()
                activity, occupancy, noise = masks(nine)
                mask = torch.from_numpy(activity)[None,None].cuda().half()
                current = inputs[:, -1:]
                gate = inputs * (1-mask) + current * mask
                past = torch.median(inputs[:, :-1].float(), dim=1, keepdim=True).values.half()
                variants = dict(baseline=inputs, current_only=current.expand_as(inputs).contiguous(),
                                gated_history=gate, past_background=past.expand_as(inputs).contiguous())
                for name, raw in variants.items():
                    y = model(raw, context)
                    if tuple(y.shape) != (1,1,3072,3840):
                        raise ValueError('Original full output geometry changed')
                    arrays[name][index] = y[0,0,::3,::3].float().cpu().numpy()
                    del y
                arrays['activity'][index] = activity
                arrays['occupancy'][index] = occupancy
                arrays['GT'][index] = read_gt(config, row, frame)
                metadata['noise'].append(noise)
                if index % 10 == 0 or index == len(frames)-1:
                    progress = dict(scene=scene, split=split, capture=capture, frame=frame,
                                    index=index+1, count=len(frames), elapsed_s=round(time.time()-started,1))
                    (REPORT / (scene+'-'+split+'-progress.json')).write_text(json.dumps(progress))
                    print('MOTION14_FRAME', json.dumps(progress), flush=True)
                del inputs, context, nine, mask, gate, current, past, variants
        for a in arrays.values():
            a.flush()
        metadata['elapsed_s'] = time.time()-started
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        (out / 'complete.json').write_text(json.dumps(metadata, indent=2))
        summary.append(metadata)
        del data, model, arrays
        gc.collect(); torch.cuda.empty_cache()
    report = dict(scene=scene, split=split, captures=summary, cache=str(target),
                  status='complete', original_ROIs_preserved=True, optimizer_steps=0,
                  baseline_checkpoint='tieddecode11 fixed500', speed_optimization=False)
    (target / 'complete.json').write_text(json.dumps(report, indent=2))
    (REPORT / (scene+'-'+split+'-prepared.json')).write_text(json.dumps(report, indent=2))
    print('MOTION14_DONE', scene, split, flush=True)

if __name__ == '__main__':
    torch.set_num_threads(2); cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--split', choices=['train','val'], required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--full-sequence', action='store_true')
    a = p.parse_args()
    run(a.scene, a.split, a.limit, a.full_sequence)
