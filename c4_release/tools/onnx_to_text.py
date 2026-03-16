"""
ONNX to Text Format Converter

Creates a simple text format that can be parsed by C4 C code.

Text Format:
  ONNX 1
  TENSORS <n>
  T <name> <ndims> <d0> <d1> ... <size> <v0> <v1> ...
  ...
  NODES <n>
  N <op> <num_in> <in0> <in1> ... <num_out> <out0> ...
  ...
  END

Usage:
    python onnx_to_text.py input.onnx output.txt
"""

import struct
import argparse
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package required")
    exit(1)

# Operation type mapping (same as onnx_to_c4.py)
OP_TYPES = {
    'Gemm': 0, 'Add': 1, 'Mul': 2, 'Sub': 3, 'Div': 4,
    'MatMul': 5, 'Relu': 6, 'Sigmoid': 7, 'Silu': 8, 'Tanh': 9,
    'Softmax': 10, 'Reshape': 11, 'Transpose': 12, 'Identity': 13,
    'Gather': 14, 'Round': 15, 'Unsqueeze': 16, 'Squeeze': 17,
    'ScatterElements': 18, 'Concat': 19, 'Slice': 20, 'Flatten': 21,
    'Floor': 22, 'Ceil': 23, 'Clip': 24, 'Neg': 25, 'Abs': 26,
    'Sqrt': 27, 'Pow': 28, 'Exp': 29, 'Log': 30, 'ReduceSum': 31,
    'ReduceMean': 32, 'ReduceMax': 33, 'Mod': 34, 'Constant': 13,
}

SCALE = 65536  # 16.16 fixed point


def float_to_fixed(val: float) -> int:
    result = int(val * SCALE)
    if result > 2147483647:
        result = 2147483647
    elif result < -2147483648:
        result = -2147483648
    return result


def convert_onnx_to_text(input_path: str, output_path: str):
    model = onnx.load(input_path)
    graph = model.graph

    tensor_names = {}
    tensors = []

    # Add initializers
    for init in graph.initializer:
        name = init.name
        tensor = numpy_helper.to_array(init)
        dims = list(tensor.shape) or [1]
        if len(dims) == 0:
            dims = [1]
            tensor = tensor.reshape(1)
        data = [float_to_fixed(float(x)) for x in tensor.flatten()]
        tensor_names[name] = len(tensors)
        tensors.append((name, dims, data))

    # Add inputs
    for inp in graph.input:
        if inp.name not in tensor_names:
            dims = []
            if inp.type.HasField('tensor_type'):
                shape = inp.type.tensor_type.shape
                for dim in shape.dim:
                    dims.append(dim.dim_value if dim.HasField('dim_value') else 1)
            if not dims:
                dims = [1]
            size = 1
            for d in dims:
                size *= d
            tensor_names[inp.name] = len(tensors)
            tensors.append((inp.name, dims, [0] * size))

    # Add node outputs
    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in tensor_names:
                dims = [1]
                if node.input and node.input[0] in tensor_names:
                    dims = list(tensors[tensor_names[node.input[0]]][1])
                size = 1
                for d in dims:
                    size *= d
                tensor_names[out_name] = len(tensors)
                tensors.append((out_name, dims, [0] * size))

    # Parse nodes
    nodes = []
    for node in graph.node:
        op_type = node.op_type
        c4_op = OP_TYPES.get(op_type, OP_TYPES.get(op_type.rstrip('_'), 13))
        inputs = [tensor_names.get(i, 0) for i in node.input]
        outputs = [tensor_names.get(o, 0) for o in node.output if o]
        nodes.append((c4_op, inputs, outputs))

    # Write text format
    with open(output_path, 'w') as f:
        f.write(f"ONNX 1\n")
        f.write(f"TENSORS {len(tensors)}\n")

        for name, dims, data in tensors:
            # T name ndims d0 d1 ... size v0 v1 ...
            dims_str = ' '.join(str(d) for d in dims)
            # Limit data output to first 1000 values for large tensors
            data_str = ' '.join(str(v) for v in data[:10000])
            f.write(f"T {name} {len(dims)} {dims_str} {len(data)} {data_str}\n")

        f.write(f"NODES {len(nodes)}\n")

        for op, inputs, outputs in nodes:
            in_str = ' '.join(str(i) for i in inputs)
            out_str = ' '.join(str(o) for o in outputs)
            f.write(f"N {op} {len(inputs)} {in_str} {len(outputs)} {out_str}\n")

        f.write("END\n")

    print(f"Converted {input_path} -> {output_path}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Nodes: {len(nodes)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input ONNX file')
    parser.add_argument('output', help='Output text file')
    args = parser.parse_args()
    convert_onnx_to_text(args.input, args.output)
