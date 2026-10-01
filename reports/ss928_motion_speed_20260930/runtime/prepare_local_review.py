"""Verify downloaded graph hashes and prepare compact, identical-frame review pictures."""
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / 'results'
    out = root / 'review_images'
    out.mkdir(exist_ok=True)
    verification = json.loads((results / 'graphs/sequence_byte_verification.json').read_text())
    records = {}
    for scene, record in verification['scenes'].items():
        path = results / 'graphs' / Path(record['graph']['path']).name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == record['graph']['sha256'], (scene, actual)
        records[scene] = {'path': str(path), 'sha256': actual}
    remote_videos = json.loads((results / 'review_video_verification.json').read_text())
    video_records = []
    families = {'history_probe': 'history', 'history_heavy': 'history',
                'reliable_evaluation': 'coarse', 'native_evaluation': 'native'}
    for record in remote_videos['videos']:
        source = Path(record['path'])
        path = results / families[source.parent.parent.name] / source.parent.name / source.name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == record['sha256'], (path, actual)
        video_records.append({'path': str(path), 'sha256': actual,
                              'decoded_frames': int(record['nb_read_frames'])})
    pictures = []
    for family, columns in [('history', 5), ('coarse', 4), ('native', 4)]:
        for path in sorted((results / family).glob('weather_*/*.png')):
            with Image.open(path) as source:
                assert source.size == (1920, 2144), (path, source.size)
                # Four-method source pictures had an unused right column.
                trimmed = source.crop((0, 0, columns * 384, 2144))
                dest = out / family / path.parent.name / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                trimmed.save(dest)
                pictures.append(str(dest.relative_to(root)))
                if path.name == 'target_072_eight_frames.png' and path.parent.name == 'weather_light':
                    # Central frame 72 is fourth in this 69..76 display.
                    trimmed.crop((0, 3 * 268, columns * 384, 4 * 268)).save(
                        out / f'{family}_light_frame072.png')
    (root / 'results/local_review_verification.json').write_text(json.dumps({
        'graphs_verified': records,
        'videos_verified': video_records,
        'pictures': pictures,
        'display_source': 'Previously generated compressed review video crops; no image metrics recomputed',
        'original_results_preserved': True,
        'packaged': False,
        'pushed': False,
    }, indent=2, ensure_ascii=False))
    print('LOCAL_REVIEW_VERIFIED', len(records), len(pictures))


if __name__ == '__main__':
    main()
