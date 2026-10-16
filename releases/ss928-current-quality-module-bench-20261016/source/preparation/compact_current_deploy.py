"""Share byte-equivalent train/smoke tensor assets while preserving both sources."""
import json
from pathlib import Path
import shutil
import numpy as np

BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'


def main():
    saved, shared = 0, []
    for path in (PACKAGE/'models').glob('*/graph_*/manifest.json'):
        folder = path.parent
        train, val = folder/'train_graph_inputs.npz', folder/'val_graph_inputs.npz'
        if not train.exists() or not val.exists():
            continue
        with np.load(train) as left, np.load(val) as right:
            equivalent = left.files == right.files and all(np.array_equal(left[name], right[name]) for name in left.files)
        if not equivalent:
            continue
        target = folder/'shared_graph_inputs.npz'
        manifest = json.loads(path.read_text())
        manifest.update(smoke_inputs=target.name, calibration_files=[target.name],
            shared_input_reason='Real train and val replay tensor arrays are exactly equal; use one stored asset. Calibration provenance remains train.',
            independent_train_val_quality_samples=False)
        archived = BASE/'capture_archive'/val.relative_to(PACKAGE)
        archived.parent.mkdir(parents=True, exist_ok=True)
        if not train.resolve().is_relative_to(PACKAGE.resolve()) or not archived.resolve().is_relative_to((BASE/'capture_archive').resolve()):
            raise ValueError('Path outside task')
        saved += val.stat().st_size
        shutil.move(str(train), str(target))
        shutil.move(str(val), str(archived))
        path.write_text(json.dumps(manifest, indent=2))
        shared.append(str(folder.relative_to(PACKAGE)))
    (PACKAGE/'SHARED_INPUT_ASSETS.json').write_text(json.dumps(dict(shared=shared, bytes_saved=saved,
        method='array equality for storage only; no model or input values changed'), indent=2))
    print('SHARED_INPUTS', len(shared), 'bytes_saved', saved, flush=True)


if __name__ == '__main__':
    main()
