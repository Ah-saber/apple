"""Print frozen tail structure for safe algebraic reparameterization."""
from front_candidates import load_front_candidate

for scene, name in (('ordinary', 'trim_t6__keep02_trained'),
                    ('special', 'trim_t6_o20')):
    model = load_front_candidate(scene, name,
        '/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V09-20260928')
    tail = model.core.model.tail
    print(scene, type(tail).__name__, tail, flush=True)
