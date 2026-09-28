"""Check all direct layouts against the frozen v0.9 FP16 complete output."""
import json
from pathlib import Path

import torch
from torch import nn

from direct_output_candidates import DirectRowsOutput
from board_candidates import GroupedRowsOutput


def main():
    torch.manual_seed(928)
    torch.set_num_threads(2)
    conv = nn.Conv2d(16, 16, 3, padding=1).half().eval()
    original = nn.Module()
    original.conv = conv
    x = torch.randn(1, 16, 32, 40).half()
    with torch.inference_mode():
        reference = GroupedRowsOutput(original, rows=32)(x)
        rows = []
        for group in (1, 2, 4, 8, 16):
            candidate = DirectRowsOutput(original, group)
            result = candidate(x)
            difference = (result.float() - reference.float()).abs()
            row = {'candidate': f'direct_g{group}', 'output_shape': list(result.shape),
                   'mean_abs_gray': float(difference.mean()),
                   'max_abs_gray': float(difference.max()),
                   'byte_equal': bool(torch.equal(result, reference))}
            assert row['byte_equal'], row
            rows.append(row)
    path = Path(__file__).with_name('direct_output_verification.json')
    path.write_text(json.dumps({'baseline': 'v0.9 rows32', 'rows': rows}, indent=2))
    print(path)


if __name__ == '__main__':
    main()
