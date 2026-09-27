import hashlib
from pathlib import Path
import torch
from load_candidate import load_candidate
from packed_front import PackedFront,PackedFullNine

def load_packed_model(path,candidate,baseline,device='cuda',output='float32',shuffle_half=True):
 ck=torch.load(path,map_location='cpu',weights_only=True)
 if ck['format']!='packed_front_v1':raise ValueError('Unexpected front format')
 if hashlib.sha256(Path(candidate).read_bytes()).hexdigest()!=ck['teacher_sha256']:raise ValueError('Wrong frozen teacher')
 c=load_candidate(candidate,baseline,device=device,remove_zero_init=True)
 f=PackedFront(ck['width']);f.load_state_dict(ck['front'],strict=True);f=f.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
 return PackedFullNine(c,f,output,shuffle_half).eval()
