"""Build an exact UINT8 output alternative to the CPU-fallback Round graph.

QuantizeLinear uses scale 1, zero point 0 and ties-to-even rounding. The
conversion still belongs to model inference; only board compilation can
establish whether the entire graph remains on the NPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def rewrite(source, destination):
    graph = onnx.load(str(source))
    old_output = graph.graph.output[0]
    source_name = old_output.name
    cast_name = 'display_gray_float32_for_quantize'
    quant_name = 'display_gray_u8_quantized'
    graph.graph.node.append(helper.make_node('Cast', [source_name], [cast_name], name='output/CastForQuantize', to=TensorProto.FLOAT))
    scale_name = 'display_gray_quantize_scale'
    zero_name = 'display_gray_quantize_zero'
    graph.graph.initializer.append(numpy_helper.from_array(np.array(1, dtype=np.float32), scale_name))
    graph.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.uint8), zero_name))
    graph.graph.node.append(helper.make_node('QuantizeLinear', [cast_name, scale_name, zero_name],
                                           [quant_name], name='output/QuantizeLinear'))
    old_output.name = quant_name
    old_output.type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(graph)
    onnx.save(graph, str(destination))


def verify(source, destination):
    original = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
    quantized = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
    feed = {}
    for input_value in original.get_inputs():
        shape = [dim if isinstance(dim, int) else 1 for dim in input_value.shape]
        rng = np.random.default_rng(928 + len(feed))
        dtype = np.float16 if input_value.type == 'tensor(float16)' else np.float32
        feed[input_value.name] = rng.uniform(0.05, 0.95, shape).astype(dtype)
    gray = original.run(None, feed)[0]
    actual = quantized.run(None, feed)[0]
    expected = np.rint(np.clip(gray.astype(np.float32), 0, 255)).astype(np.uint8)
    return {'source': source.name, 'candidate': destination.name,
            'output_shape': list(actual.shape), 'output_dtype': str(actual.dtype),
            'exact_byte_equality': bool(np.array_equal(actual, expected)),
            'different_pixels': int(np.count_nonzero(actual != expected))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in ('ordinary', 'special'):
        for small in (False, True):
            suffix = '_small' if small else ''
            source = args.source_dir / f'{scene}_rows32{suffix}.onnx'
            destination = args.output_dir / f'{scene}_rows32_quantizelinear_u8{suffix}.onnx'
            rewrite(source, destination)
            if small:
                row = verify(source, destination)
                assert row['exact_byte_equality'], row
                rows.append(row)
                print(row, flush=True)
    (args.output_dir / 'verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
