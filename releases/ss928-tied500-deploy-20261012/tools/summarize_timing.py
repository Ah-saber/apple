"""Summarize measured pure NPU time separately from SDK synchronization time."""
import argparse,csv,json
from pathlib import Path
import numpy as np
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--warmup-rows',type=int,default=0);a=p.parse_args()
    groups={}
    with a.csv.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):groups.setdefault((r['scene'],r['role']),[]).append(r)
    records=[]
    for (scene,role),rows in groups.items():
        measured=rows[a.warmup_rows:];result=dict(scene=scene,role=role,iterations=len(measured),warmup_rows_removed=a.warmup_rows)
        for col in ['pure_npu_ms','sdk_sync_ms']:
            values=[float(r[col]) for r in measured if r.get(col,'').strip()]
            result[col]=dict(samples=len(values),mean=float(np.mean(values)),P95=float(np.percentile(values,95)),max=max(values)) if values else None
        npu=result['pure_npu_ms'];result['target_16_7ms_met']=bool(npu and npu['samples']==len(measured) and npu['mean']<=16.7 and npu['P95']<=16.7)
        records.append(result)
    result=dict(records=records,boundary='Complete graph pure NPU readings supplied by deployment runner; SDK synchronization remains separate. Missing pure NPU data cannot pass.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
