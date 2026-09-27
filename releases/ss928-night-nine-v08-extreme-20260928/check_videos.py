import json,subprocess
from pathlib import Path
r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');rows=[]
for f in sorted((r/'videos').glob('*.mp4')):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(f)]));s=d['streams'][0];expected=60 if f.name.startswith('ordinary_') else 120;assert s['width']==5120 and s['height']==1080 and int(s['nb_frames'])==expected and s['avg_frame_rate']=='12/1';rows.append({'file':str(f.relative_to(r)),**s,'full_sensor_panels':True,'SDK_video':False})
(r/'full_video_checks.json').write_text(json.dumps({'count':len(rows),'videos':rows},indent=2));print('VIDEOS_VERIFIED',len(rows))
