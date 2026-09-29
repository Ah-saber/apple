"""Append direct UINT8 cast to complete reference-reordered candidates."""
import argparse
import hashlib
import json
from pathlib import Path

from build_byte_candidates import add_cast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--lowref-dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scene in ('ordinary', 'special'):
        for layout in ('rows32', 'direct6', 'native_nearest', 'native_deconv',
                       'native_separable'):
            for small in (False, True):
                suffix = '_small' if small else ''
                source = args.source_dir / (
                    f'{scene}_reference_after12_{layout}{suffix}.onnx')
                target = args.out / (
                    f'{scene}_reference_after12_{layout}_cast_u8{suffix}.onnx')
                add_cast(source, target)
                manifest.append({'name': target.name,
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'source': source.name, 'NPU_verified': False,
                    'rounding_on_SS928_verified': False})
    if args.lowref_dir is not None:
        for layout in ('rows32', 'direct6', 'native_nearest',
                       'native_separable'):
            for small in (False, True):
                suffix = '_small' if small else ''
                source = args.lowref_dir / (
                    f'special_lowref_k5_{layout}{suffix}.onnx')
                target = args.out / (
                    f'special_lowref_k5_{layout}_cast_u8{suffix}.onnx')
                add_cast(source, target)
                manifest.append({'name': target.name,
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'source': source.name, 'NPU_verified': False,
                    'rounding_on_SS928_verified': False})
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
