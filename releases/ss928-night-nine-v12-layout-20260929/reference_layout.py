"""Move the 12->16 pointwise reference projection across bilinear resize."""
from front_candidates import load_front_candidate
from output_candidates import OutputLayout, prepare_inputs


def load_candidate(scene, layout, output_layout, run_dir, device='cuda'):
    if layout not in ('before16', 'after12'):
        raise ValueError(layout)
    stem = ('trim_t6__keep02_trained' if scene == 'ordinary'
            else 'trim_t6_o20')
    model = load_front_candidate(scene, stem, run_dir, device=device)
    model.output = OutputLayout(model.output, output_layout).to(device=device).eval()
    if layout == 'after12':
        model.project_after_resize = True
    return model.eval()
