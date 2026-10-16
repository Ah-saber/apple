"""Preserve initial interface cases before taking an established-history sample."""
import json
from pathlib import Path
import shutil

BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'

for path in (PACKAGE/'models').glob('*/graph_*/manifest.json'):
    manifest = json.loads(path.read_text())
    if not manifest.get('boundary_precision_changes'):
        continue
    folder = path.parent
    archive = BASE/'initial_fp32_cases'/folder.relative_to(PACKAGE)
    if not folder.resolve().is_relative_to(PACKAGE.resolve()) or not archive.resolve().is_relative_to((BASE/'initial_fp32_cases').resolve()):
        raise ValueError('Path outside task')
    archive.mkdir(parents=True, exist_ok=True)
    for source in folder.iterdir():
        if source.is_file():
            shutil.copy2(source, archive/source.name)
    shutil.move(str(path), str(archive/'initial_manifest.json'))
    for filename in ['val_inputs.npz', 'train_inputs.npz', 'val_pytorch_output.npz', 'train_pytorch_output.npz']:
        source = BASE/'capture_archive'/folder.relative_to(PACKAGE)/filename
        if source.exists():
            shutil.move(str(source), str(archive/filename))
    print('INITIAL_CASE_PRESERVED', folder.name)
