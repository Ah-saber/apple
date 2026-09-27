import subprocess,json
from pathlib import Path
p='/data/zhangbenzhuang/miniconda3/envs/test/bin/python'
r=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');c=r/'code_snapshot'
cases=['cal_preserve_mean6_body3_native4_alignpixel', 'cal_preserve_sum6_body3_native4_alignpixel', 'preserve_raw6_body3_native4_alignpixel', 'preserve_mean6_body3', 'preserve_sum6_body3', 'cal_preserve_mean6_body3_native4_alignpixel_kernel7', 'cal_preserve_mean6_body3_native4_alignpixel_layout16', 'cal_preserve_mean6_body3_native4_alignpixel_layout5', 'cal_preserve_mean6_body3_native4_alignpixel_f32io', 'cal_preserve_mean6_body3_native4_alignpixel_outdeconv', 'cal_preserve_mean6_body3_native4_alignpixel_outcascade', 'cal_preserve_mean6_body3_native4_alignpixel_refpad16_meanfirst', 'cal_preserve_mean6_body3_native4_alignpixel_block16_full', 'cal_preserve_mean6_body3_native4_alignpixel_block16_split_full', 'cal_preserve_mean6_body3_native4_alignpixel_block16', 'quarter_w32_3x3_body1_gtweak', 'cal_quarter_w32_3x3_body1_gtweak', 'quarter_w32_3x3_body1_gtweak_deconv', 'quarter_w32_3x3_body1_gtweak_cascade', 'quarter_w32_3x3_body1_gtweak_refsplit', 'cal_graph_quarter_w32_3x3_body1_gtweak_refsplit', 'quarter_w32_3x3_body1_stage1_fusedtail_refsplit', 'quarter_w32_3x3_body1_stage1_deconv_fusedtail', 'quarter_w32_3x3_body1_stage1_cascade_fusedtail']
for scene in ('ordinary','special'):
 jobs=[('export_candidates.py',['--tag','supplement_export','--cases',*cases]),('export_reference_vectors.py',['--tag','supplement_manifest','--cases',*cases]),('check_small_onnx.py',['--tag','supplement_small_ort','--cases',cases[0],cases[3],cases[6],cases[7],cases[8],cases[9],cases[10],cases[16],cases[18],cases[19],cases[20]])]
 for variant in [cases[0],cases[15]]:jobs.append(('make_full_videos.py',['--baseline','body3_quantsearch','--variant',variant]))
 for name,args in jobs:
  with (r/(scene+'_finish_'+name+'_'+str(len(args))+'.log')).open('w') as log:subprocess.run([p,str(c/name),'--scene',scene,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
print('SUPPLEMENT_COMPLETE',flush=True)
