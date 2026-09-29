"""Build a broader day train cache without touching validation or test frames."""
import argparse
import sys
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--crops',type=int,default=4)
    a=p.parse_args()
    sys.path[:0]=[str(a.code/'src'),str(Path(__file__).parent)]
    from ir_sr.training import dataset_for_config
    from train_quarter_student import cache_data
    torch.set_num_threads(2)
    state=torch.load(a.config_checkpoint,map_location='cpu',weights_only=False)
    train=dataset_for_config(state['config'],'train')
    a.out.mkdir(parents=True,exist_ok=True)
    values,metadata=cache_data(train,None,state['config'],['day_normal'],
                                a.out,929,a.rows,a.crops)
    print('train only',len(metadata),'pairs',flush=True)


if __name__=='__main__':main()
