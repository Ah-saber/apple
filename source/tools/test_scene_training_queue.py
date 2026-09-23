"""Safety checks for the new serial queue; CPU-only, no training or GPU allocation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_scene_training_queue import validate_jobs, require_complete_training


class QueueSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        code = Path(__file__).resolve().parents[1]
        self.plan = json.loads((code / 'configs/train/day_weather_queue_d1.json').read_text())
        for i, job in enumerate(self.plan['jobs']):
            src = code / job['config']
            dst = self.root / job['config']
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            job['run'] = str(self.root / ('run%d' % i))
            job['finish'] = str(self.root / ('finish%d' % i))
        self.output = self.root / 'queue'

    def tearDown(self):
        self.temp.cleanup()

    def test_existing_output_refused_before_any_launch(self):
        validate_jobs(self.plan, self.root, self.output)
        Path(self.plan['jobs'][1]['run']).mkdir()
        with self.assertRaises(FileExistsError):
            validate_jobs(self.plan, self.root, self.output)

    def test_duplicate_output_refused(self):
        self.plan['jobs'][1]['run'] = self.plan['jobs'][0]['run']
        with self.assertRaises(ValueError):
            validate_jobs(self.plan, self.root, self.output)

    def test_scene_and_recipe_drift_refused(self):
        path = self.root / self.plan['jobs'][2]['config']
        config = json.loads(path.read_text())
        for field, value in [('scene_ids', ['night_special']), ('max_steps', 120000),
                             ('train_crop_hr', [1152, 1152])]:
            changed = copy.deepcopy(config)
            changed[field] = value
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                validate_jobs(self.plan, self.root, self.output)
        path.write_text(json.dumps(config))

    def test_short_training_or_hard_timeout_stops_queue(self):
        run = Path(self.plan['jobs'][0]['run']); run.mkdir()
        for steps, reason, hard, valid in [(200000, 'max_steps', False, True),
                (140000, 'training_time_budget', False, False), (200000, 'max_steps', True, False)]:
            (run / 'exit.json').write_text(json.dumps({'exit_code': 0, 'hard_timeout': hard}))
            (run / 'completed.json').write_text(json.dumps({'status': 'completed', 'steps': steps, 'reason': reason}))
            if valid:
                require_complete_training(run, 200000)
            else:
                with self.assertRaises(RuntimeError):
                    require_complete_training(run, 200000)


if __name__ == '__main__':
    unittest.main()
