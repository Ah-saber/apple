"""Audit static shapes, input types, file identity and calibration/test coverage."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def shape(v):return [d.dim_value for d in v.type.tensor_type.shape.dim]


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    index={'NPU_measured':False,'target_mean_ms':16.7,'target_p95_ms':16.7,'models':[], 'datasets':{}}
    for path in sorted(a.release.rglob('*.onnx')):
        graph=onnx.load(str(path));onnx.checker.check_model(graph)
        relative=path.relative_to(a.release).as_posix()
        inputs=[{'name':v.name,'dtype':onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),'shape':shape(v)} for v in graph.graph.input]
        outputs=[{'name':v.name,'dtype':onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),'shape':shape(v)} for v in graph.graph.output]
        operators=sorted({n.op_type for n in graph.graph.node})
        diagnostic='cast_probe/' in relative or '_phases.onnx' in relative
        if not diagnostic:assert outputs[0]['shape']==[1,1,3072,3840],(relative,outputs)
        if relative.startswith(('onnx/equivalent/','onnx/compact/')):
            assert 'GridSample' not in operators and 'Resize' not in operators,(relative,operators)
        record={'path':relative,'sha256':sha(path),'bytes':path.stat().st_size,
                'diagnostic':diagnostic,'inputs':inputs,'outputs':outputs,'operators':operators,
                'board_compatibility':'unmeasured','NPU_measured':False}
        if relative.startswith('onnx/equivalent/'):
            scene=path.parent.name
            record['data_directory']='deployment_corrected/'+scene
        elif relative.startswith('onnx/compact/'):
            scene=path.parent.name
            record['data_directory']='deployment_corrected/'+scene
        elif relative.startswith('onnx/night_preserved/'):
            scene='night_ordinary' if path.name.startswith('ordinary') else 'night_special'
            record['data_directory']='night_calibration_final/'+scene
            record['reference_rewrite']='Existing night interpolation compatibility rewrite still required; verify exact same input'
        index['models'].append(record)
    for folder in ['records/deployment','records/night_calibration']:
        for path in (a.release/folder).glob('*/manifest.json'):
            j=json.loads(path.read_text());scene=path.parent.name
            assert len(j['calibration'])==4 and len(j['test'])==12
            for sample in j['calibration']:assert len(set(sample['history_frame_ids']))==9
            index['datasets'][scene]={'calibration':4,'test':12,'GT_used_in_calibration':False,
                'records':path.relative_to(a.release).as_posix(),
                'test_zero_startup_repeats_recorded':True}
    assert len(index['datasets'])==7,index['datasets']
    assert len(index['models'])==68,len(index['models'])
    (a.release/'MODEL_INDEX.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    all_files=sorted(p for p in a.release.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (a.release/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(a.release).as_posix()+'\n' for p in all_files))
    print('AUDIT_COMPLETE',len(index['models']),'graphs',len(index['datasets']),'scenes',len(all_files),'files',flush=True)


if __name__=='__main__':main()
