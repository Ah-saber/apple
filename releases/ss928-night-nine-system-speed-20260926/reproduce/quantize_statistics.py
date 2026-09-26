"""Constant-signal gain under explicit hypothetical weight quantizers."""
import json
from pathlib import Path
import numpy as np
rows=[]
for name,coeff in [('old6_equal',[1/6]*6),('old6_balanced',[120/727]*5+[127/727]),('recent3_equal',[1/3]*3),('recent3_balanced',[123/373]*2+[127/373])]:
 coeff=np.asarray(coeff,dtype=np.float16).astype(np.float64)
 for maximum in (127,255):
  for scope in ('all_stats_max1','per_output'):
   scale=(1 if scope=='all_stats_max1' else coeff.max())/maximum;q=np.rint(coeff/scale);gain=float((q*scale).sum());rows.append({'weights':name,'max_integer':maximum,'scope':scope,'quantized_gain_constant_signal':gain,'difference_after_x64_at_raw_0.5':(1-gain)*32})
Path(__file__).with_name('statistics_quantization_simulation.json').write_text(json.dumps({'assumed_quantizers_only':True,'SDK_quantizer_known':False,'NPU_verified':False,'controls':rows},indent=2));print(json.dumps(rows,indent=2))
