"""Invoke the deployment team's SS928 ATC entry using the delivered static interface."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--role',choices=['retained_original','tied500'],required=True)
    p.add_argument('--atc',required=True,help='Existing deployment atc executable or its environment wrapper')
    p.add_argument('--image-list',required=True,help='Training calibration argument in the installed SDK image_list format; use the four prepared train samples')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--compile-mode',default='1');p.add_argument('--print-only',action='store_true');a=p.parse_args()
    folder=ROOT/'models'/a.scene;meta=json.loads((folder/'manifest.json').read_text());graph=meta['graphs'][a.role]
    args=[a.atc,'--framework=5','--soc_version=SS928V100','--npu_arch=V101','--model='+str((folder/graph['file']).resolve()),
        '--input_shape='+';'.join(s['name']+':'+','.join(map(str,s['shape'])) for s in graph['inputs']),
        '--input_type='+';'.join(s['name']+':FP16' for s in graph['inputs']),'--output_type=FP16',
        '--image_list='+a.image_list,'--compile_mode='+a.compile_mode,'--online_model_type=3','--log_level=1','--output='+str(a.out.resolve()/'model')]
    print(json.dumps(args),flush=True)
    if not a.print_only:
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'command.json').write_text(json.dumps(args,indent=2))
        with (a.out/'conversion.log').open('wb') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
