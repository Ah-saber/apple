"""Diagnostic ONLY; the goal-qualified graph performs unpack internally."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--dtype',choices=['float16','float32'],required=True);p.add_argument('--output',required=True);a=p.parse_args();x=np.fromfile(a.input,dtype=np.dtype(a.dtype));assert x.size==3072*3840;full=x.reshape(1,16,192,3840).transpose(0,2,1,3).reshape(1,1,3072,3840);np.save(a.output,full);print(json.dumps({'shape':list(full.shape),'dtype':str(full.dtype),'CPU_unpack_done':True,'graph_is_diagnostic_only_not_full_goal_timing':True}))
