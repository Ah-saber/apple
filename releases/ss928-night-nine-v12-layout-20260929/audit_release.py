"""Check interfaces, static dimensions and hashes for board screening graphs."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto


DIRECTORIES = ('source_output', 'source_reference', 'source_lowref',
               'source_byte', 'source_combined_cast', 'output_micrographs')


def shape(value):
    dimensions = value.type.tensor_type.shape.dim
    assert all(d.HasField('dim_value') for d in dimensions)
    return [d.dim_value for d in dimensions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for folder in DIRECTORIES:
        for path in sorted((args.root / folder).glob('*.onnx')):
            if path.stem.startswith('cast_'):
                continue
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            small = '_small' in path.stem
            micro = folder == 'output_micrographs'
            inputs = list(model.graph.input)
            output, = model.graph.output
            if micro:
                expected_input = [1, 16, 64, 64] if small else [1, 16, 512, 640]
                assert len(inputs) == 1 and shape(inputs[0]) == expected_input, path
            else:
                expected_input = [1, 9, 128, 128] if small else [1, 9, 1024, 1280]
                assert len(inputs) == 2 and shape(inputs[0]) == expected_input, path
                assert shape(inputs[1]) == [1, 1, 64, 64], path
            expected_output = [1, 1, 384, 384] if small else [1, 1, 3072, 3840]
            assert shape(output) == expected_output, path
            expected_type = (TensorProto.UINT8 if folder in
                             ('source_byte', 'source_combined_cast')
                             else TensorProto.FLOAT16)
            assert output.type.tensor_type.elem_type == expected_type, path
            entries.append({'file': str(path.relative_to(args.root)),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'inputs': [shape(value) for value in inputs],
                'output': expected_output,
                'output_type': TensorProto.DataType.Name(expected_type),
                'nodes': len(model.graph.node)})
    assert entries
    result = {'graphs': len(entries), 'counts': {
        directory: sum(row['file'].startswith(directory + '/') for row in entries)
        for directory in DIRECTORIES}, 'entries': entries,
        'NPU_verified': False}
    (args.root / 'graph_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'graphs': result['graphs'], 'counts': result['counts']}))


if __name__ == '__main__':
    main()
