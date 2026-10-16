"""Link GT regions across frames. Fixed texture and isolated GT noise are excluded."""
from collections import defaultdict
import math
import numpy as np


class GTTracks:
    def __init__(self):
        self.tracks = []
        self.active = {}

    @staticmethod
    def center(item):
        if 'centroid_yx' in item:
            return tuple(item['centroid_yx'])
        y, x, h, w = item['bbox_yxhw']
        return y+h/2, x+w/2

    def valid(self, track):
        if len(track['items']) < 3:
            return False
        points = np.asarray(track['points'])
        displacement = float(np.linalg.norm(points.max(0)-points.min(0)))
        path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        net = float(np.linalg.norm(points[-1]-points[0]))
        return displacement >= 1.5 and net/max(path, 1e-6) >= .35

    def update(self, frame, items):
        buckets = defaultdict(list)
        for ident in list(self.active):
            track = self.tracks[ident]
            if frame-track['last_frame'] > 2:
                del self.active[ident]
                continue
            y, x = track['points'][-1]
            buckets[(int(y//16), int(x//16))].append(ident)
        proposals = []
        for i, item in enumerate(items):
            y, x = self.center(item)
            cy, cx = int(y//16), int(x//16)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    for ident in buckets[(cy+dy, cx+dx)]:
                        track = self.tracks[ident]
                        last = track['items'][-1]
                        if last['polarity'] != item['polarity']:
                            continue
                        ratio = item['target_pixels']/last['target_pixels']
                        if not .25 <= ratio <= 4:
                            continue
                        py, px = track['points'][-1]
                        distance = math.hypot(y-py, x-px)
                        if distance > 16*(frame-track['last_frame']):
                            continue
                        cost = distance+abs(math.log(ratio))
                        proposals.append((cost, i, ident))
        used_items, used_tracks = set(), set()
        assignments = {}
        for _, i, ident in sorted(proposals):
            if i in used_items or ident in used_tracks:
                continue
            assignments[i] = ident
            used_items.add(i)
            used_tracks.add(ident)
        confirmed = []
        for i, item in enumerate(items):
            ident = assignments.get(i)
            if ident is None:
                ident = len(self.tracks)
                self.tracks.append({'items': [], 'points': [], 'last_frame': frame})
            track = self.tracks[ident]
            track['items'].append(item)
            track['points'].append(self.center(item))
            track['last_frame'] = frame
            item['gt_track_id'] = ident
            self.active[ident] = True
            if self.valid(track):
                confirmed.append(i)
        return confirmed

    def selected(self):
        return [item for track in self.tracks if self.valid(track) for item in track['items']]

    def summary(self):
        valid = [t for t in self.tracks if self.valid(t)]
        return {'tracks': len(valid), 'observations': sum(len(t['items']) for t in valid),
                'rule': 'GT-only nearest region match; same polarity; area ratio .25..4; gap <=2 frames; >=3 observations; centroid range >=1.5 native pixels; net/path displacement >=.35',
                'semantic_person_labels': False}


def save_gallery(metric, directory):
    import json
    from PIL import Image, ImageDraw
    if metric.worst is None:
        return
    record = metric.worst
    imgs = record['images']
    gt = imgs['GT']
    low, high = float(gt.min()), float(gt.max())
    high = max(high, low+1)
    for mode in ['raw_gray', 'common_gt_scale']:
        canvas = Image.new('RGB', (256*len(imgs), 288), '#181818')
        draw = ImageDraw.Draw(canvas)
        for col, (name, pixels) in enumerate(imgs.items()):
            if mode == 'common_gt_scale':
                pixels = (pixels-low)*255/(high-low)
            tile = Image.fromarray(np.rint(pixels.clip(0, 255)).astype(np.uint8)).convert('RGB').resize((256, 256), Image.Resampling.NEAREST)
            canvas.paste(tile, (256*col, 32))
            draw.text((256*col+5, 8), name, fill='white')
        canvas.save(directory/('tracked_weak_gap_'+mode+'.png'))
    (directory/'tracked_weak_gap.json').write_text(json.dumps({'target': record['record'], 'common_scale_gt_min_max': [low, high]}, indent=2))
