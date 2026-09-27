"""Summarize measured board samples; validate full-model status against graph matrix."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--output',required=True);p.add_argument('--matrix',default=str(Path(__file__).parent/'DEPLOYMENT-MATRIX.json'));a=p.parse_args();matrix=json.loads(Path(a.matrix).read_text());known={Path(r['file']).name:r for r in matrix['graphs']};rows={};flags={}
with Path(a.samples).open(newline='') as f:
 for r in csv.DictReader(f):
  if r.get('phase','measure')!='measure':continue
  graph=Path(r['graph']).name
  if graph not in known:raise ValueError('Graph missing from frozen deployment matrix: '+graph)
  value=float(r['time_ms']);assert np.isfinite(value) and value>=0
  full=r['full_model'].lower() in ('true','1','yes')
  if full!=known[graph]['full_model']:raise ValueError('Incorrect full-model flag for '+graph)
  rows.setdefault(graph,[]).append(value);flags[graph]=full
assert rows
results=[]
for graph,values in rows.items():
 mean=float(np.mean(values));p95=float(np.percentile(values,95));results.append({'graph':graph,'graph_sha256':known[graph]['sha256'],'count':len(values),'mean_ms':mean,'p95_ms':p95,'full_model':flags[graph],'timing_target_met':bool(flags[graph] and len(values)>=200 and mean<=16.7 and p95<=16.7),'quality_passed':None,'SDK_artifact_hash_verified':None})
report={'target_mean_ms':16.7,'target_p95_ms':16.7,'minimum_measured_samples':200,'measurement_source':'provided actual board CSV; synchronized measurement must be verified separately','module_times_not_summed':True,'quality_identity_input_preprocessing_and_synchronization_require_separate_verification':True,'results':sorted(results,key=lambda r:(not r['full_model'],r['mean_ms']))};out=Path(a.output)
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
