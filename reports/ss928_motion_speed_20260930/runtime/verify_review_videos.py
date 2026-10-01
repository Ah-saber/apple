"""Decode-count all nine complete experimental comparison videos."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-MOTION-SPEED-20260930')
paths = []
for family in ['history_probe', 'history_heavy', 'reliable_evaluation', 'native_evaluation']:
    paths.extend(sorted((root / family).glob('weather_*/full120_history_intervention.mp4')))
assert len(paths) == 9, len(paths)
records = []
for path in paths:
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
        '-of', 'json', str(path)]))['streams'][0]
    expected_width = 6400 if path.parent.parent.name.startswith('history') else 5120
    assert (probe['width'], probe['height']) == (expected_width, 1080), (path, probe)
    assert int(probe['nb_read_frames']) == 120 and probe['avg_frame_rate'] == '12/1', (path, probe)
    records.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **probe})
    print('VIDEO_VERIFIED', str(path.relative_to(root)), flush=True)
(root / 'review_video_verification.json').write_text(json.dumps({
    'videos': records, 'total_decoded_frames': 1080,
    'display_fps_is_capture_fps': False, 'manual_all_frames_reviewed': False,
    'packaged': False, 'pushed': False,
}, indent=2))
print('ALL_NINE_REVIEW_VIDEOS_VERIFIED', flush=True)
