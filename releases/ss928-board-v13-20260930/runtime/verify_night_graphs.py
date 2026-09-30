"""Verify the preserved night ONNX paths with their actual mixed input types."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from run_compact import ROOT, sha
from prepare_deployment import ONNX_SITE
from verify_release import ort_run, delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(ONNX_SITE))
    v12=ROOT/'runs/SS928-BOARD-V12-20260929'
    report={'NPU_measured':False,'provider':'CPUExecutionProvider','scenes':{}}
    for scene,folder,name in [('ordinary','source_reference','ordinary_reference_after12_rows32.onnx'),
                              ('special','source_lowref','special_lowref_k5_rows32.onnx')]:
        directory=a.root/'night_calibration_final'/('night_'+scene)
        sample=np.load(directory/'test_00.npz',allow_pickle=False)
        graph=v12/folder/name
        record=delta(ort_run(graph,sample),sample['pc_native_gray'])
        record.update({'graph':str(graph),'sha256':sha(graph),'sample':str(directory/'test_00.npz'),
                       'comparison':'exact preserved ONNX CPU against FP16 source PyTorch GPU; mixed input types preserved'})
        report['scenes'][scene]=record
        print('ORT_NIGHT',scene,record,flush=True)
    (a.root/'verification/night_exact_graph_verification.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
