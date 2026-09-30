"""Freeze this iteration; keep bulk calibration separate from the Git release."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from run_compact import ROOT, CODE, sha


def copy_file(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
    assert sha(source)==sha(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    a.release.mkdir(parents=True,exist_ok=False)
    for path in (a.root/'runtime').glob('*.py'):
        copy_file(path,a.release/'runtime'/path.name)
    for path in (CODE/'src/ir_sr').glob('*.py'):
        copy_file(path,a.release/'source/ir_sr'/path.name)
    night_code=ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925'
    for path in (night_code/'src/ir_sr').glob('*.py'):
        copy_file(path,a.release/'night_source/ir_sr'/path.name)
    legacy=ROOT/'runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS'
    for name in ['half_student.py','train_quarter_student.py','export_half_board.py']:
        copy_file(legacy/name,a.release/'runtime/legacy'/name)
    identity={'source_git_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),
              'source_status':subprocess.check_output(['git','-C',str(CODE),'status','--short'],text=True),
              'python':sys.version,'executable':sys.executable,'bulk_data':[], 'experiments':{}}
    identity['night_source_git_commit']=subprocess.check_output(['git','-C',str(night_code),'rev-parse','HEAD'],text=True).strip()
    identity['night_source_status']=subprocess.check_output(['git','-C',str(night_code),'status','--short'],text=True)
    assert identity['source_git_commit']=='0c1f674d7b5f1d1aed8e75ac64082c97e0dedc9e'
    for name in ['day','day_fused','light_fused','heavy_fused','light_temporal','heavy_temporal']:
        run=a.root/name
        if not (run/'evaluation.json').exists():continue
        copy_file(run/'best.pt',a.release/'models'/(name+'.pt'))
        for fname in ['training.json','evaluation.json']:
            copy_file(run/fname,a.release/'records'/name/fname)
        identity['experiments'][name]=json.loads((run/'evaluation.json').read_text())
        for path in (run/'onnx').rglob('*') if (run/'onnx').exists() else []:
            if path.is_file():copy_file(path,a.release/'onnx/compact'/name/path.relative_to(run/'onnx'))
        for path in run.glob('*_full120.mp4'):
            copy_file(path,a.release/('videos/rejected' if name in ['day','day_fused'] else 'videos/candidates')/name/path.name)
    for path in (a.root/'deployment_corrected').rglob('*.onnx'):
        copy_file(path,a.release/'onnx/equivalent'/path.relative_to(a.root/'deployment_corrected'))
    for path in (a.root/'deployment_corrected').rglob('*.json'):
        copy_file(path,a.release/'records/deployment'/path.relative_to(a.root/'deployment_corrected'))
    for path in (a.root/'cast_probe').glob('*'):
        if path.is_file():copy_file(path,a.release/'cast_probe'/path.name)
    for path in (a.root/'verification').glob('*.json'):
        copy_file(path,a.release/'records/verification'/path.name)
    v12=ROOT/'runs/SS928-BOARD-V12-20260929'
    for folder,name in [('source_reference','ordinary_reference_after12_rows32.onnx'),
                        ('source_lowref','special_lowref_k5_rows32.onnx')]:
        copy_file(v12/folder/name,a.release/'onnx/night_preserved'/name)
    for name in ['final_temporal.json','final_fused.json']:
        if (a.root/name).exists():copy_file(a.root/name,a.release/'records'/name)
    for group in ['HALF-DAY-NINE-TEMPORAL','HALF-LIGHT-TEMPORAL','HALF-HEAVY-TEMPORAL','HALF-DAY-RAWSKIP']:
        name='best.pt' if group.startswith('HALF-DAY') else 'average05.pt'
        path=ROOT/'runs'/('SS928-FIVE-SCENE-'+group+'-20260929')/name
        copy_file(path,a.release/'models/baseline'/(group.lower()+'.pt'))
    for path in a.root.glob('*.log'):
        copy_file(path,a.release/'records/logs'/path.name)
    if (a.root/'night_calibration_final').exists():
        for path in (a.root/'night_calibration_final').rglob('*.json'):
            copy_file(path,a.release/'records/night_calibration'/path.relative_to(a.root/'night_calibration_final'))
    # Preserve every referenced normalization index independently of the server path.
    for dataset in ['deployment_corrected','night_calibration_final']:
        for path in (a.root/dataset).glob('*/config.json'):
            config=json.loads(path.read_text())
            for key,value in config.items():
                if 'index' in key and isinstance(value,str) and Path(value).is_file():
                    copy_file(Path(value),a.release/'normalization'/(sha(Path(value))+'_'+Path(value).name))
        if dataset=='night_calibration_final':
            for path in (a.root/dataset).glob('*/manifest.json'):
                for key,value in json.loads(path.read_text())['config'].items():
                    if 'index' in key and isinstance(value,str) and Path(value).is_file():
                        copy_file(Path(value),a.release/'normalization'/(sha(Path(value))+'_'+Path(value).name))
    # Include the original lossless GT files in the separate evidence bundle.
    for dataset in ['deployment_corrected','night_calibration_final']:
        root=a.root/dataset
        for manifest in root.glob('*/manifest.json') if root.exists() else []:
            j=json.loads(manifest.read_text())
            for i,entry in enumerate(j['test']):
                row=entry['sample']
                source=ROOT/'data'/row['target']['path']
                dest=manifest.parent/'gt'/f'{i:02d}_frame_{row["frame_id"]:06d}.png'
                copy_file(source,dest)
    bundle=a.root/'SS928-V13-REAL-CALIBRATION-AND-GT-20260930.tar'
    with tarfile.open(bundle,'w') as tf:
        for dataset in ['deployment_corrected','night_calibration_final']:
            folder=a.root/dataset
            if not folder.exists():continue
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix!='.onnx':
                    tf.add(path,arcname=Path(dataset)/path.relative_to(folder))
    identity['bulk_data'].append({'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size,
                                 'contains':'real train/test inputs, uint16 histories, exact GT PNGs and PC layer/output references'})
    (a.release/'IDENTITY.json').write_text(json.dumps(identity,indent=2))
    print('RELEASE',a.release,'BULK',bundle,identity['bulk_data'],flush=True)


if __name__=='__main__':main()
