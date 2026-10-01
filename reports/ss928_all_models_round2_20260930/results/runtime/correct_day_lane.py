"""Offline-only correction of the day bridge mask; original caches and metrics are retained."""
import json,sys
from pathlib import Path
import cv2,numpy as np
R=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-ALL-MODELS-ROUND2-20260930')
polygon=[[350,379],[450,406],[600,438],[800,486],[1000,539],[1279,607],[1279,621],[1000,552],[800,505],[600,456],[450,421],[350,394]]
sources={'adaptive':'adaptive_nonzero_day','residual':'residual_nonzero_eval_day','three':'half3_eval_day','one':'half1_eval_day'}
for label,folder in sources.items():
    old=R/folder/'day_normal';out=R/('day_lane_'+label)/'day_normal';out.mkdir(parents=True,exist_ok=False)
    saved=np.load(old/'road_signals_and_masks.npz');mask=np.zeros(saved['GT'].shape[-2:],np.uint8);cv2.fillPoly(mask,[np.array(polygon)],1);mask=mask.astype(bool)
    values={}
    for key in saved.files:
        if key=='GT_current_mask':continue
        value=saved[key];values[key]=value-np.median(value[:,mask],axis=1)[:,None,None]
    np.savez_compressed(out/'road_signals_and_masks.npz',**values)
    (out/'full120_history_intervention.mp4').symlink_to(old/'full120_history_intervention.mp4')
    (out/'target_region.json').write_text(json.dumps({'roi_y':0,'roi_x':0,'polygon':polygon,'valid_native_tlhw':[2,0,1020,1278],'metric_border_pixels':3,'selection':'straight driving lane traced on same-scene GT frame60, x>=350; excludes upper buildings and parked cars; previous polygon rejected after picture review','source_float_cache':str(old/'road_signals_and_masks.npz'),'offline_recenter':'subtract per-frame median in corrected lane; no model or input changes'}))
    print('DAY_LANE_CORRECTED',label,flush=True)
