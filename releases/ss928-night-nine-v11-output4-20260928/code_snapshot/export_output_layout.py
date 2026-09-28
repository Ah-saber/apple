"""Export complete explicit-repeat layout graphs for NPU compilation tests."""
import argparse, fcntl, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

from output_layout_candidates import load_output_candidate as load_candidate, prepare_inputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['ordinary','special'],required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--v09-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.v09_run/'code_snapshot/tools'))
    sys.path.insert(0,'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
    import onnx
    from materialize_aliases import materialize
    with (a.v09_run.parent/'TASK-019-gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        frame=20 if a.scene=='ordinary' else 60
        vec=np.load(a.v09_run/'v08/test_vectors'/f'{a.scene}_frame_{frame}.npz')
        x=torch.from_numpy(vec['nine_raw']).cuda();ctx=torch.from_numpy(vec['reference_thumb']).cuda()
        rows=[];baseline=None
        for case in a.cases:
            model=load_candidate(a.scene,case,a.v09_run).eval()
            xx,cc=prepare_inputs(model,x,ctx)
            with torch.inference_mode():value=model(xx,cc).cpu().numpy()
            assert list(value.shape)==[1,1,3072,3840] and np.isfinite(value).all()
            if baseline is None:baseline=value
            delta=np.abs(value.astype(np.float32)-baseline.astype(np.float32))
            np.savez_compressed(a.output/f'{a.scene}_{case}_reference.npz',output=value)
            for small in (False,True):
                h,w=(64,96) if small else (1024,1280)
                ax=torch.full((1,9,h,w),.3,device='cuda')
                ac=torch.full((1,1,64,64),.3,device='cuda')
                ax,ac=prepare_inputs(model,ax,ac)
                path=a.output/f'{a.scene}_{case}{"_small" if small else ""}.onnx'
                torch.onnx.export(model,(ax,ac),str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
                graph,_=materialize(onnx.load(path));onnx.checker.check_model(graph);onnx.save(graph,path)
                shape=[d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
                assert shape==[1,1,h*3,w*3],shape
                row={'scene':a.scene,'case':case,'small':small,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':shape,'mean_abs_gray_vs_first':float(delta.mean()),'max_abs_gray_vs_first':float(delta.max()),'NPU_verified':False}
                rows.append(row);print(row,flush=True)
            del model
            torch.cuda.empty_cache()
        (a.output/f'{a.scene}_export_manifest.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
