"""Alternate AB/BA calls to compare complete outputs under a shared GPU load."""
import time
import numpy as np
import torch

def paired_timing(models,inputs,warm=30,steps=150):
    names=list(models)
    values={name:dict(gpu_event=[],full_synchronized_wall=[]) for name in names}
    start=torch.cuda.Event(enable_timing=True); finish=torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for i in range(warm):
            for name in names: models[name](*inputs[i%len(inputs)])
        torch.cuda.synchronize()
        for i in range(steps):
            for name in (names if i%2==0 else list(reversed(names))):
                before=time.perf_counter_ns(); start.record()
                result=models[name](*inputs[i%len(inputs)])
                finish.record(); finish.synchronize()
                values[name]['full_synchronized_wall'].append((time.perf_counter_ns()-before)/1e6)
                values[name]['gpu_event'].append(start.elapsed_time(finish))
                assert result.shape==(1,1,3072,3840) and result.dtype==torch.float16
    return {name:{stage:dict(mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95))) for stage,samples in stages.items()} for name,stages in values.items()}
