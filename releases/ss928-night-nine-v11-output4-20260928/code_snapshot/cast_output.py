"""Alternative UINT8 graph using FP16 Add(0.5) and Cast, with documented tie change."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def rewrite(source, destination):
    graph = onnx.load(str(source))
    output = graph.graph.output[0]
    old = output.name
    half_name = 'display_gray_add_half'
    byte_name = 'display_gray_cast_u8'
    scalar = 'display_gray_half_scalar'
    graph.graph.initializer.append(numpy_helper.from_array(np.array(.5, dtype=np.float16), scalar))
    graph.graph.node.append(helper.make_node('Add', [old, scalar], [half_name], name='output/AddHalf'))
    graph.graph.node.append(helper.make_node('Cast', [half_name], [byte_name],
                                           name='output/CastToU8', to=TensorProto.UINT8))
    output.name = byte_name
    output.type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(graph)
    onnx.save(graph, str(destination))


def verify(source, destination):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    original = ort.InferenceSession(str(source), opts, providers=['CPUExecutionProvider'])
    revised = ort.InferenceSession(str(destination), opts, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(928)
    feed = {}
    for meta in original.get_inputs():
        shape = [v if isinstance(v, int) else 1 for v in meta.shape]
        dtype = np.float16 if meta.type == 'tensor(float16)' else np.float32
        feed[meta.name] = rng.uniform(.05, .95, shape).astype(dtype)
    source_output = original.run(None, feed)[0]
    actual = revised.run(None, feed)[0]
    expected = np.trunc((source_output + np.float16(.5)).astype(np.float32)).astype(np.uint8)
    even = np.rint(source_output.astype(np.float32)).astype(np.uint8)
    row = {'source': source.name, 'candidate': destination.name,
           'exact_half_up': bool(np.array_equal(actual, expected)),
           'different_from_even_pixels': int(np.count_nonzero(actual != even)),
           'fraction_different_from_even': float(np.mean(actual != even)),
           'mean_bias_gray': float((actual.astype(np.int16) - even.astype(np.int16)).mean())}
    assert row['exact_half_up'], row
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in ('ordinary', 'special'):
        for small in (False, True):
            suffix = '_small' if small else ''
            source = a.source_dir / f'{scene}_rows32{suffix}.onnx'
            target = a.output_dir / f'{scene}_rows32_addhalf_cast_u8{suffix}.onnx'
            rewrite(source, target)
            if small:
                item = verify(source, target)
                rows.append(item)
                print(item, flush=True)
    (a.output_dir / 'addhalf_verification.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
