"""Freeze RAW-only frame compensation and auxiliary-target coordinate metadata."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ir_sr.training import dataset_for_config,validate_data_lock,atomic_json
from ir_sr.sequence_normalization import calibrate_sequence,digest
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
c=json.loads(args.config.read_text());root=Path(c['data_root']);code=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
start=time.perf_counter();lock=validate_data_lock(root,code);chosen={}
for split in ['train','val','test']:
 for r in dataset_for_config(c,split).records:
  key=r['domain']+'/'+r['sequence_id']
  if key in chosen:assert chosen[key]['split']==split
  chosen.setdefault(key,r)
mi=json.loads((Path(c['middle_gt_root'])/'index.json').read_text());mid={r['key']:r for r in mi['sequences']};rows=[]
for key,r in sorted(chosen.items()):
 t=time.perf_counter();path=root/r['raw']['path'];sha=digest(path);h=time.perf_counter();raw=np.load(path,mmap_mode='r');roi=r.get('train_roi_tlhw',[0,0,1024,1280]);top,left,height,width=roi
 parameters,segments=calibrate_sequence(raw[:,top:top+height,left:left+width]);cal_seconds=time.perf_counter()-h
 if r['split']=='train':
  m=mid[key];assert m['roi_tlhw']==roi and m['configuration']['branch']=='weather-v6' and m['configuration']['profile']['coarse_sigma']==0
  assert m['effective_processing']['segments']==[[s['begin'],s['end']] for s in segments]
 assert digest(path)==sha
 rows.append({'key':key,'raw_path':r['raw']['path'],'raw_sha256':sha,'frames':len(raw),'roi_tlhw':roi,'split':r['split'],'parameters':parameters,'segments':segments,'calibration_seconds':cal_seconds,'wall_seconds':time.perf_counter()-t})
 print(key,roi,flush=True)
report={'status':'complete','kind':'raw_frame_center_segment_v1','data_lock_sha256':lock,'middle_gt_index_sha256':digest(Path(c['middle_gt_root'])/'index.json'),'protocol':{'input':'(RAW-current_center-common_residual-corrected_reference_low)/segment_span','middle':'(denoised_DN-segment_median_center-corrected_reference_low)/segment_span; no repeated current-frame drift removal','loss':'aux error * span/original109.56366230459803, then weight0.1','regions':'All statistics inside allowed training ROI; heldout sequences use own full RAW only','mode':'offline complete segment; future RAW required; no GT used in parameter fitting'},'sequences':rows,'wall_seconds':time.perf_counter()-start}
atomic_json(args.output/'index.json',report);print('COMPLETE',digest(args.output/'index.json'),report['wall_seconds'],flush=True)
