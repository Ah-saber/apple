"""Preserve generated FP64 graph versions before exporting explicit FP32 interfaces."""
import json
from pathlib import Path
import shutil

BASE = Path('/tmp/rawir-current-deploy-20261016')
PACKAGE = BASE/'package'

for folder in (PACKAGE/'models').glob('*/graph_*'):
    path = folder/'manifest.json'
    if not path.exists():
        continue
    manifest = json.loads(path.read_text())
    if not any(value['dtype'] == 'DOUBLE' for value in manifest['inputs']+manifest['outputs']):
        continue
    archive = BASE/'fp64_graph_archive'/folder.relative_to(PACKAGE)
    if not folder.resolve().is_relative_to(PACKAGE.resolve()) or not archive.resolve().is_relative_to((BASE/'fp64_graph_archive').resolve()):
        raise ValueError('Path outside current task')
    archive.mkdir(parents=True, exist_ok=True)
    for source in folder.iterdir():
        if source.is_file():
            shutil.copy2(source, archive/source.name)
    shutil.move(str(path), str(archive/'original_manifest.json'))
    print('ARCHIVED_FP64_INTERFACE', folder.name)
