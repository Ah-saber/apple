"""Load independent frozen teacher and trained trajectory student artifacts."""
import hashlib
from pathlib import Path
import torch
from load_candidate import load_candidate
from trajectory_student import TrajectoryStudent,StudentFullNine

def load_student_model(student_path,candidate_path,baseline_path,device='cuda',execution='source'):
 ck=torch.load(student_path,map_location='cpu',weights_only=True)
 if ck['format']!='trajectory_student_v1':raise ValueError('Unexpected student format')
 if hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest()!=ck['teacher_sha256']:raise ValueError('Wrong frozen teacher')
 core=load_candidate(candidate_path,baseline_path,device=device,remove_zero_init=True)
 s=TrajectoryStudent(ck['width'],ck.get('quarter_stem',False));s.load_state_dict(ck['student'],strict=True)
 s=s.to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
 model=StudentFullNine(core,s).eval()
 if execution=='compiled':return torch.compile(model,fullgraph=True,options={'triton.cudagraphs':False})
 if execution!='source':raise ValueError('Unexpected execution mode')
 return model
