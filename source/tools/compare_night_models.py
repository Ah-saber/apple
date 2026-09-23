"""Compare one night-only model with a mixed B0 selected by the same night validation rule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image, ImageDraw
import torch
from ir_sr.model import RT4KSRB0, to_deploy
from ir_sr.training import atomic_json, dataset_for_config, evaluate, sha, uint8_image
from finalize_training import native_artifacts


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.run/'config.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==config['gpu_uuid']
    assert config['scene_ids']==['night_ordinary','night_special']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    old_config=json.loads((args.baseline/'config.json').read_text())
    for key in ['channels','blocks','scale','dataset','data_root','seed','metrics']:
        assert config[key]==old_config[key],key
    old_rows=[json.loads(s) for s in (args.baseline/'metrics.jsonl').read_text().splitlines()]
    old_val=[r for r in old_rows if r['split']=='val']
    scenes=config['scene_ids']
    chosen=max(old_val,key=lambda r:sum(r['scene_metrics'][s]['psnr'] for s in scenes)/len(scenes))
    assert chosen['step']==config.get('comparison_baseline_step',70000), 'Frozen validation-selected baseline changed'
    index=json.loads((args.baseline/'checkpoint_index.json').read_text())
    entry=next(r for r in index['checkpoints'] if r['step']==chosen['step'])
    old_path=args.baseline/entry['path']
    assert sha(old_path)==entry['sha256']
    old_model=RT4KSRB0(config['channels'],config['blocks']).cuda().eval()
    old_model.load_state_dict(torch.load(old_path,map_location='cpu',weights_only=False)['model'])
    old_deployed=to_deploy(old_model)
    new_result=json.loads((args.run/'result.json').read_text())
    new_weights=args.run/'artifacts/b0_deploy.pt'
    assert sha(new_weights)==new_result['deployment_weight_sha256']
    new_model=RT4KSRB0(config['channels'],config['blocks'],deploy=True).cuda().eval()
    new_model.load_state_dict(torch.load(new_weights,map_location='cpu',weights_only=False)['model'])
    comparison={'baseline_selection':'mean PSNR of both night validation scenes only; all historical checkpoints',
                'baseline_checkpoint':entry,'night_checkpoint':new_result['selected_checkpoint'],'splits':{},
                'baseline_label':config.get('comparison_baseline_label','Mixed B0'),
                'candidate_label':config.get('comparison_candidate_label','Night-only B0'),
                'training_exposure_note':'Compare complete recipes: training membership, patch and step budget may differ; not a single-variable ablation.',
                'training_config_difference':{k:{'baseline':old_config.get(k),'candidate':config.get(k)}
                                              for k in sorted(set(old_config)|set(config)) if old_config.get(k)!=config.get(k)},
                'SS928':'not verified'}
    for split in ('val','test'):
        old=evaluate(old_model,config['data_root'],split,'cuda',args.output/('baseline_'+split),entry['step'],True,scenes)
        fresh=json.loads((args.run/'selected_best'/split/'metrics.json').read_text())
        assert [r['sample_id'] for r in old['images']]==[r['sample_id'] for r in fresh['images']]
        prior=next((r for r in old_rows if r['step']==entry['step'] and r['split']==split), None)
        # Validation checkpoints need not coincide with periodic test steps.
        # Test is freshly evaluated at the validation-selected weight, never reselected.
        if split=='val':
            assert prior is not None, 'Selected baseline validation evidence is missing'
        values={}
        for scene in scenes:
            a,b=old['scene_metrics'][scene],fresh['scene_metrics'][scene]
            if prior is not None:
                assert abs(a['psnr']-prior['scene_metrics'][scene]['psnr'])<1e-4
            values[scene]={'mixed':a,'night':b,'delta_psnr':b['psnr']-a['psnr'],'delta_ssim':b['ssim']-a['ssim']}
        comparison['splits'][split]={'scenes':values,'mixed_macro':old['macro'],'night_macro':fresh['macro'],
                                    'historical_baseline_score_available':prior is not None}
    native_artifacts(old_deployed,config,args.output)
    ds=dataset_for_config(config,'val');norm=ds.recipe['raw_normalization']
    chosen_rows={}
    for row in ds.records:chosen_rows.setdefault(row['scene_id'],row)
    visuals=args.output/'comparison_visuals';visuals.mkdir()
    for scene,row in sorted(chosen_rows.items()):
        with Image.open(args.run/'native_qualitative'/(scene+'_teacher_bicubic_reference.png')) as im:teacher=im.copy()
        with Image.open(args.output/'native_qualitative'/(scene+'_sr_x3.png')) as im:old_image=im.copy()
        with Image.open(args.run/'native_qualitative'/(scene+'_sr_x3.png')) as im:new_image=im.copy()
        images=[teacher,old_image,new_image]
        labels=['Teacher bicubic: not HR truth',comparison['baseline_label'],comparison['candidate_label']]
        view=Image.new('RGB',(2400,674),'#202020');draw=ImageDraw.Draw(view)
        for i,(im,label) in enumerate(zip(images,labels)):
            view.paste(im.resize((800,640),Image.Resampling.LANCZOS),(i*800,30))
            draw.text((i*800+8,8),label,fill='white')
        view.save(visuals/(scene+'_overview.jpg'),quality=94)
        for region,box in [('building',(1536,1536,2304,2304)),('top',(1536,0,2304,768))]:
            view=Image.new('RGB',(2304,802),'#202020');draw=ImageDraw.Draw(view)
            for i,(im,label) in enumerate(zip(images,labels)):
                view.paste(im.crop(box),(i*768,30));draw.text((i*768+8,8),label,fill='white')
            view.save(visuals/(scene+'_'+region+'.jpg'),quality=95)
        raw_cache=np.load(Path(config['data_root'])/row['raw']['path'],mmap_mode='r')
        first=min(len(raw_cache)-12,len(raw_cache)//2)
        command=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','gray','-s','3840x1024','-r','6','-i','-',
                 '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-y',str(visuals/(scene+'_12frames_comparison.mp4'))]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for frame_id in range(first,first+12):
                value=(raw_cache[frame_id].astype(np.float32)-norm['offset'])/norm['scale']
                x=torch.from_numpy(value)[None,None].cuda()
                panels=[]
                with Image.open((Path(config['data_root'])/row['target']['path']).with_name('%06d.png'%frame_id)) as im:
                    panels.append(np.array(im))
                for model in (old_deployed,new_model):
                    panels.append(np.asarray(Image.fromarray(uint8_image(model(x))).resize((1280,1024),Image.Resampling.LANCZOS)))
                process.stdin.write(np.concatenate(panels,axis=1).tobytes())
        finally:process.stdin.close()
        assert process.wait()==0
    comparison['visual_layout']='teacher bicubic / baseline step%d / candidate selected best; native qualitative, no true HR'%entry['step']
    comparison['legacy_field_roles']={'mixed':'baseline','night':'candidate'}
    atomic_json(args.output/'comparison.json',comparison)
    print(json.dumps(comparison),flush=True)


if __name__=='__main__':main()
