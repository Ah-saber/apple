"""Build direct FP16->UINT8 casts and tiny rounding probes for SS928."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


VALUES = np.array([0, .25, .49, .5, .51, .75, 1.25, 1.49,
                   1.5, 1.51, 1.75, 2.49, 2.5, 2.51, 127.5,
                   254.5], dtype=np.float16).reshape(1, 1, 1, 16)


def add_cast(source, target):
    graph = onnx.load(str(source))
    original = graph.graph.output[0]
    name = original.name
    output = 'display_gray_cast_u8'
    graph.graph.node.append(helper.make_node('Cast', [name], [output],
        name='output/CastDirectU8', to=TensorProto.UINT8))
    original.name = output
    original.type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))


def probe_graph(path, add_half):
    x = helper.make_tensor_value_info('gray_fp16', TensorProto.FLOAT16,
                                      [1, 1, 1, 16])
    y = helper.make_tensor_value_info('gray_u8', TensorProto.UINT8,
                                      [1, 1, 1, 16])
    nodes = []
    initializers = []
    value = 'gray_fp16'
    if add_half:
        initializers.append(numpy_helper.from_array(
            np.array(.5, dtype=np.float16), 'half'))
        nodes.append(helper.make_node('Add', ['gray_fp16', 'half'],
                                      ['gray_plus_half'], name='probe/AddHalf'))
        value = 'gray_plus_half'
    nodes.append(helper.make_node('Cast', [value], ['gray_u8'],
                                  name='probe/CastU8', to=TensorProto.UINT8))
    graph = helper.make_graph(nodes, path.stem, [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    report = {'full_graphs': [], 'rounding_probes': []}
    for scene in ('ordinary', 'special'):
        for layout in ('rows32', 'direct6', 'native_deconv',
                       'native_nearest', 'native_separable'):
            for small in (False, True):
                suffix = '_small' if small else ''
                source = args.source_dir / f'{scene}_{layout}{suffix}.onnx'
                target = args.out / f'{scene}_{layout}_cast_u8{suffix}.onnx'
                add_cast(source, target)
                report['full_graphs'].append({'source': source.name,
                    'candidate': target.name, 'NPU_verified': False})
    for add_half in (False, True):
        name = 'cast_addhalf_probe.onnx' if add_half else 'cast_direct_probe.onnx'
        path = args.out / name
        probe_graph(path, add_half)
        session = ort.InferenceSession(str(path), options,
                                       providers=['CPUExecutionProvider'])
        actual = session.run(None, {'gray_fp16': VALUES})[0]
        source = VALUES + np.float16(.5) if add_half else VALUES
        expected = np.trunc(source.astype(np.float32)).astype(np.uint8)
        assert np.array_equal(actual, expected)
        report['rounding_probes'].append({'graph': name,
            'add_half': add_half, 'input_float16': VALUES.flatten().tolist(),
            'onnx_runtime_uint8': actual.flatten().tolist(),
            'NPU_verified': False})
        print(report['rounding_probes'][-1], flush=True)
    np.savez(args.out / 'cast_probe_input.npz', gray_fp16=VALUES)
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
