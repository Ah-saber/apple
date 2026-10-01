"""Load all saved architecture options and compare with independently saved v13 ONNX."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
from run_compact import GROUPS,setup,sha
from compact_model import BoardCompact
from prepare_deployment import ONNX_SITE


def load_group(group):
    state=None
    if group!='day':
        state=torch.load(V13/(group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)
        GROUPS[group].update(state['spec'])
    values=setup(group)
    if state is not None:
        compact=values[0];compact.load_state_dict(state['model'])
        for key in ['width','depth','raw_skip','late_reference']:
            assert getattr(compact,key)==state['spec'][key],(key,getattr(compact,key),state['spec'][key])
    return values


def independent_control(group,compact):
    if group=='day':return {'kind':'preserved half architecture; quarter unused'}
    sys.path.insert(0,str(ONNX_SITE));import onnxruntime as ort
    scene=GROUPS[group]['scenes'][0]
    directory=V13/'deployment_corrected'/scene
    if not (directory/'test_00.npz').exists():directory=V13/'deployment'/scene
    sample=np.load(directory/'test_00.npz');box=torch.tensor([np.load(directory/'test_00.npz')['context_box'].tolist()],device='cuda') if 'context_box' in sample.files else None
    if box is None:
        import json
        manifest=__import__('json').loads((directory/'manifest.json').read_text())
        box=torch.tensor([manifest['context_box']],device='cuda')
    graph=V13/(group+'_temporal')/'onnx'/scene/(scene+'_compact_fp32_phases.onnx')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(graph),opts,providers=['CPUExecutionProvider'])
    inputs={k:sample[k].astype(np.float32) for k in ['nine_raw','reference_thumb']}
    expected=session.run(None,inputs)[0]
    model=BoardCompact(compact,box,layout='phases').cuda().float().eval()
    with torch.inference_mode():actual=model(*(torch.from_numpy(inputs[k]).cuda() for k in ['nine_raw','reference_thumb'])).cpu().numpy()
    error=np.abs(actual-expected);record={'old_graph':str(graph),'old_graph_sha256':sha(graph),'mean_gray':float(error.mean()),'max_gray':float(error.max()),
        'late_reference':compact.late_reference,'weights_sha256':sha(V13/(group+'_temporal')/'best.pt')}
    assert record['max_gray']<.01,record
    print('INDEPENDENT_V13_BASELINE',group,record,flush=True)
    return record
