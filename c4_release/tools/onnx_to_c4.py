"""
ONNX to C4 Binary Format Converter

Converts standard ONNX models to the simplified binary format used by
the C4-compatible ONNX runtime (onnx_runtime_c4.c).

Modes:
  1. Single ONNX -> c4onnx v1 (original):
       python onnx_to_c4.py input.onnx output.c4onnx

  2. Directory of ONNX -> c4onnx v3 with subgraphs:
       python onnx_to_c4.py --to-c4onnx-v3 models/sparse_onnx/ -o output.c4onnx

     Reads individual .onnx files (b2n, n2b, nib_add, etc.), converts
     their weights to sparse COO 16.16 fixed-point, maps Mul(x,scalar)
     to OP_SCALE, and emits a single v3 file with named subgraphs.
"""

import struct
import argparse
import os
import glob
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package required. Install with: pip install onnx")
    exit(1)


# Operation type mapping (v1 format)
# Must match onnx_runtime_c4.c definitions
OP_TYPES = {
    'Gemm': 0,
    'Add': 1,
    'Mul': 2,
    'Sub': 3,
    'Div': 4,
    'MatMul': 5,
    'Relu': 6,
    'Sigmoid': 7,
    'Silu': 8,
    'Tanh': 9,
    'Softmax': 10,
    'Reshape': 11,
    'Transpose': 12,
    'Identity': 13,
    # Extended ops (need to add to C4 runtime)
    'Gather': 14,
    'Round': 15,
    'Unsqueeze': 16,
    'Squeeze': 17,
    'ScatterElements': 18,
    'Concat': 19,
    'Slice': 20,
    'Flatten': 21,
    'Floor': 22,
    'Ceil': 23,
    'Clip': 24,
    'Neg': 25,
    'Abs': 26,
    'Sqrt': 27,
    'Pow': 28,
    'Exp': 29,
    'Log': 30,
    'ReduceSum': 31,
    'ReduceMean': 32,
    'ReduceMax': 33,
    'Mod': 34,
    'Fmod': 34,
    # Common variations/aliases
    'Add_': 1,
    'Mul_': 2,
    'MatMul_': 5,
    'Constant': 13,  # Constants are just identity
}

# v3 subgraph op types (must match onnx_runtime_c4.c / export_vm_weights.py)
OP_MATMUL_V3 = 5
OP_SOFTMAX_V3 = 10
OP_CONCAT_V3 = 14
OP_SLICE_V3 = 15
OP_SCALE_V3 = 16

# Fixed-point scale (16.16 format)
SCALE = 65536


def float_to_fixed(val: float) -> int:
    """Convert float32 to 16.16 fixed-point integer."""
    result = int(val * SCALE)
    # Clamp to 32-bit signed integer range
    if result > 2147483647:
        result = 2147483647
    elif result < -2147483648:
        result = -2147483648
    return result


def write_int(f, val: int):
    """Write 32-bit little-endian integer."""
    # Handle negative numbers with two's complement
    if val < 0:
        val = val + 0x100000000
    f.write(struct.pack('<I', val & 0xFFFFFFFF))


def convert_onnx_to_c4(input_path: str, output_path: str, verbose: bool = False):
    """Convert ONNX model to C4 binary format."""

    # Load ONNX model
    model = onnx.load(input_path)
    graph = model.graph

    # Build tensor name -> index mapping
    tensor_names = {}
    tensors = []  # List of (name, dims, data)

    # 1. Add initializers (model weights)
    for init in graph.initializer:
        name = init.name
        tensor = numpy_helper.to_array(init)

        # Flatten complex shapes if needed
        dims = list(tensor.shape)
        if len(dims) == 0:
            dims = [1]
            tensor = tensor.reshape(1)

        # Convert to fixed-point
        data = [float_to_fixed(float(x)) for x in tensor.flatten()]

        tensor_names[name] = len(tensors)
        tensors.append((name, dims, data))

        if verbose:
            print(f"  Tensor {len(tensors)-1}: {name} {dims}")

    # 2. Add graph inputs (that aren't initializers)
    for inp in graph.input:
        if inp.name not in tensor_names:
            # Get shape from type info
            dims = []
            if inp.type.HasField('tensor_type'):
                shape = inp.type.tensor_type.shape
                for dim in shape.dim:
                    if dim.HasField('dim_value'):
                        dims.append(dim.dim_value)
                    else:
                        dims.append(1)  # Unknown dims default to 1

            if len(dims) == 0:
                dims = [1]

            # Initialize with zeros
            size = 1
            for d in dims:
                size *= d
            data = [0] * size

            tensor_names[inp.name] = len(tensors)
            tensors.append((inp.name, dims, data))

            if verbose:
                print(f"  Input {len(tensors)-1}: {inp.name} {dims}")

    # 3. Add intermediate tensors (node outputs)
    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in tensor_names:
                # Infer shape from inputs if possible, otherwise use [1]
                dims = [1]

                # Try to infer from first input
                if len(node.input) > 0 and node.input[0] in tensor_names:
                    in_idx = tensor_names[node.input[0]]
                    dims = list(tensors[in_idx][1])

                size = 1
                for d in dims:
                    size *= d
                data = [0] * size

                tensor_names[out_name] = len(tensors)
                tensors.append((out_name, dims, data))

                if verbose:
                    print(f"  Intermediate {len(tensors)-1}: {out_name} {dims}")

    # 4. Parse nodes
    nodes = []  # List of (op_type, inputs, outputs)

    for node in graph.node:
        op_type = node.op_type

        # Map to C4 op type
        if op_type not in OP_TYPES:
            # Try common variations
            if op_type.endswith('_'):
                op_type_base = op_type[:-1]
                if op_type_base in OP_TYPES:
                    op_type = op_type_base

        if op_type not in OP_TYPES:
            print(f"Warning: Unknown op type '{node.op_type}', using Identity")
            c4_op = OP_TYPES['Identity']
        else:
            c4_op = OP_TYPES[op_type]

        # Get input/output tensor indices
        inputs = []
        for inp in node.input:
            if inp in tensor_names:
                inputs.append(tensor_names[inp])
            else:
                print(f"Warning: Unknown input tensor '{inp}'")
                inputs.append(0)

        outputs = []
        for out in node.output:
            if out in tensor_names:
                outputs.append(tensor_names[out])
            else:
                print(f"Warning: Unknown output tensor '{out}'")
                outputs.append(0)

        nodes.append((c4_op, inputs, outputs))

        if verbose:
            print(f"  Node: {node.op_type} -> op={c4_op}, in={inputs}, out={outputs}")

    # 5. Write binary output
    with open(output_path, 'wb') as f:
        # Header
        write_int(f, 0x584E4E4F)  # "ONNX" magic
        write_int(f, 1)            # Version
        write_int(f, len(tensors))
        write_int(f, len(nodes))

        # Tensors
        for name, dims, data in tensors:
            name_bytes = name.encode('utf-8')[:63]  # Max 63 chars
            write_int(f, len(name_bytes))
            f.write(name_bytes)

            write_int(f, len(dims))
            for d in dims:
                write_int(f, d)

            write_int(f, 0)  # data_type (float)
            write_int(f, len(data))
            for val in data:
                write_int(f, val)

        # Nodes
        for op_type, inputs, outputs in nodes:
            write_int(f, op_type)
            write_int(f, len(inputs))
            for inp in inputs:
                write_int(f, inp)
            write_int(f, len(outputs))
            for out in outputs:
                write_int(f, out)

    print(f"Converted {input_path} -> {output_path}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Nodes: {len(nodes)}")

    return True


def tensor_to_sparse_coo(flat_data, threshold=0):
    """Convert flat fixed-point data to COO format.
    Returns (indices, values, nnz) or None if not worth sparsifying."""
    indices = []
    values = []
    for i, v in enumerate(flat_data):
        if v != 0:
            indices.append(i)
            values.append(v)
    nnz = len(indices)
    if nnz * 2 < len(flat_data):
        return indices, values, nnz
    return None


def is_scalar_tensor(name, init_map):
    """Check if a tensor name refers to a scalar initializer."""
    if name not in init_map:
        return False
    arr = init_map[name]
    return arr.size == 1


def get_scalar_value(name, init_map):
    """Get scalar value from an initializer."""
    return float(init_map[name].flatten()[0])


def parse_onnx_for_v3(model_path, verbose=False):
    """Parse a standard ONNX model and extract weights + subgraph structure.

    Returns (weight_tensors, subgraph_def) where:
      weight_tensors: list of (name, numpy_array) for weight matrices
      subgraph_def: (name, num_inputs, num_outputs, num_temps, temp_sizes, nodes)
    """
    model = onnx.load(model_path)
    graph = model.graph
    sg_name = os.path.splitext(os.path.basename(model_path))[0]

    # Collect initializers
    init_map = {}
    for init in graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    # Identify graph inputs (non-initializer)
    graph_inputs = []
    for inp in graph.input:
        if inp.name not in init_map:
            dims = []
            if inp.type.HasField('tensor_type'):
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        dims.append(dim.dim_value)
            graph_inputs.append((inp.name, dims))

    # Identify graph outputs
    graph_outputs = [(out.name, []) for out in graph.output]

    num_inputs = len(graph_inputs)
    num_outputs = len(graph_outputs)

    # Build name->local_ref mapping
    # Inputs: -1, -2, ...
    # Temps: -(num_inputs+1), -(num_inputs+2), ...
    # Outputs: -(num_inputs+num_temps+1), ...
    # Global tensors (weights): positive index

    # First pass: identify weight tensors we need (2D matrices, not slice params)
    weight_names = []
    for init_name, arr in init_map.items():
        if arr.ndim >= 2:
            weight_names.append(init_name)

    # Map weight names to global indices (will be assigned by caller)
    # For now, use placeholder names

    # Second pass: figure out temps and output mapping
    input_names = {name: -(i + 1) for i, (name, _) in enumerate(graph_inputs)}
    output_names = {name for name, _ in graph_outputs}

    # Collect intermediate tensor names (node outputs that aren't graph outputs)
    temp_names = {}
    temp_sizes_list = []
    for node in graph.node:
        for out_name in node.output:
            if out_name and out_name not in output_names and out_name not in init_map:
                if out_name not in temp_names:
                    temp_idx = len(temp_names)
                    temp_names[out_name] = -(num_inputs + temp_idx + 1)
                    # Infer size from context
                    temp_sizes_list.append(0)  # Will be filled later

    num_temps = len(temp_names)

    # Map output names
    output_ref_map = {}
    for i, (out_name, _) in enumerate(graph_outputs):
        output_ref_map[out_name] = -(num_inputs + num_temps + i + 1)

    # Infer temp sizes from the graph
    # We need to trace through ops to figure out sizes
    name_to_size = {}
    for name, dims in graph_inputs:
        size = 1
        for d in dims:
            size *= d
        name_to_size[name] = size
    for name, arr in init_map.items():
        name_to_size[name] = arr.size

    for node in graph.node:
        if node.op_type == "Concat":
            total = sum(name_to_size.get(inp, 0) for inp in node.input)
            for out_name in node.output:
                name_to_size[out_name] = total
        elif node.op_type == "MatMul":
            # out size = second dim of second input
            w_name = node.input[1]
            if w_name in init_map:
                name_to_size[node.output[0]] = init_map[w_name].shape[-1]
        elif node.op_type in ("Mul", "Softmax"):
            # Same size as first input
            for out_name in node.output:
                name_to_size[out_name] = name_to_size.get(node.input[0], 0)
        elif node.op_type == "Slice":
            # Parse start/end from initializers
            if len(node.input) >= 3:
                start_name = node.input[1]
                end_name = node.input[2]
                if start_name in init_map and end_name in init_map:
                    start = int(init_map[start_name].flatten()[0])
                    end = int(init_map[end_name].flatten()[0])
                    name_to_size[node.output[0]] = end - start

    # Update temp sizes
    for name, ref in temp_names.items():
        idx = -(ref) - num_inputs - 1
        temp_sizes_list[idx] = name_to_size.get(name, 0)

    def get_ref(name):
        """Get local reference for a tensor name."""
        if name in input_names:
            return input_names[name]
        if name in temp_names:
            return temp_names[name]
        if name in output_ref_map:
            return output_ref_map[name]
        # It's a weight — return as string placeholder
        return name

    # Third pass: build node list
    nodes = []
    weight_tensors = []
    weight_name_to_global = {}

    for node in graph.node:
        if node.op_type == "Concat":
            inputs = [get_ref(inp) for inp in node.input]
            outputs = [get_ref(node.output[0])]
            nodes.append((OP_CONCAT_V3, inputs, outputs))

        elif node.op_type == "MatMul":
            in_ref = get_ref(node.input[0])
            w_name = node.input[1]
            out_ref = get_ref(node.output[0])
            # Weight reference is a string — will be resolved later
            nodes.append((OP_MATMUL_V3, [in_ref, w_name], [out_ref]))

        elif node.op_type == "Mul":
            # Check if this is Mul(x, scalar) -> OP_SCALE
            in0 = node.input[0]
            in1 = node.input[1]
            if is_scalar_tensor(in1, init_map):
                scalar_val = get_scalar_value(in1, init_map)
                scalar_fp = float_to_fixed(scalar_val)
                in_ref = get_ref(in0)
                out_ref = get_ref(node.output[0])
                nodes.append((OP_SCALE_V3, [in_ref, scalar_fp], [out_ref]))
            elif is_scalar_tensor(in0, init_map):
                scalar_val = get_scalar_value(in0, init_map)
                scalar_fp = float_to_fixed(scalar_val)
                in_ref = get_ref(in1)
                out_ref = get_ref(node.output[0])
                nodes.append((OP_SCALE_V3, [in_ref, scalar_fp], [out_ref]))
            else:
                # Regular element-wise mul (not expected in our models)
                nodes.append((OP_TYPES['Mul'], [get_ref(in0), get_ref(in1)],
                             [get_ref(node.output[0])]))

        elif node.op_type == "Softmax":
            in_ref = get_ref(node.input[0])
            out_ref = get_ref(node.output[0])
            nodes.append((OP_SOFTMAX_V3, [in_ref], [out_ref]))

        elif node.op_type == "Slice":
            # Extract start/end from initializer inputs
            in_ref = get_ref(node.input[0])
            out_ref = get_ref(node.output[0])
            start = 0
            end = 0
            if len(node.input) >= 3:
                start_name = node.input[1]
                end_name = node.input[2]
                if start_name in init_map:
                    start = int(init_map[start_name].flatten()[0])
                if end_name in init_map:
                    end = int(init_map[end_name].flatten()[0])
            nodes.append((OP_SLICE_V3, [in_ref, start, end], [out_ref]))

        else:
            if verbose:
                print(f"  Warning: skipping op {node.op_type} in {sg_name}")

    # Collect weight tensors and resolve string references
    for i, (op, inputs, outputs) in enumerate(nodes):
        resolved_inputs = []
        for inp in inputs:
            if isinstance(inp, str) and inp in init_map:
                # This is a weight tensor reference
                if inp not in weight_name_to_global:
                    arr = init_map[inp]
                    weight_name_to_global[inp] = len(weight_tensors)
                    weight_tensors.append((f"{sg_name}_{inp}", arr))
                resolved_inputs.append(weight_name_to_global[inp])
            else:
                resolved_inputs.append(inp)
        nodes[i] = (op, resolved_inputs, outputs)

    subgraph = (sg_name, num_inputs, num_outputs, num_temps, temp_sizes_list, nodes)

    if verbose:
        print(f"  Parsed {sg_name}: {num_inputs} inputs, {num_outputs} outputs, "
              f"{num_temps} temps, {len(nodes)} nodes, {len(weight_tensors)} weights")

    return weight_tensors, subgraph


def build_auxiliary_tensors():
    """Build shift tables and constants needed by the bundled runtime.

    These don't have ONNX models — they're simple computed tables.
    Returns list of (name, dims, fixed_data, sparse_info).
    """
    tensors = []

    # Shift tables: [5][16][16]
    shl_result = np.zeros((5, 16, 16))
    shl_overflow = np.zeros((5, 16, 16))
    shr_result = np.zeros((5, 16, 16))
    shr_underflow = np.zeros((5, 16, 16))

    for shift in range(5):
        for val in range(16):
            res = (val << shift) & 0xF
            over = (val >> (4 - shift)) & 0xF if shift > 0 else 0
            shl_result[shift, val, res] = 1.0
            shl_overflow[shift, val, over] = 1.0
            res = (val >> shift) & 0xF
            under = (val << (4 - shift)) & 0xF if shift > 0 else 0
            shr_result[shift, val, res] = 1.0
            shr_underflow[shift, val, under] = 1.0

    for name, arr in [("shl_result", shl_result), ("shl_overflow", shl_overflow),
                      ("shr_result", shr_result), ("shr_underflow", shr_underflow)]:
        dims = list(arr.shape)
        flat = arr.flatten().astype(float)
        data = [float_to_fixed(v) for v in flat]
        sparse = tensor_to_sparse_coo(data)
        tensors.append((name, dims, data, sparse))

    # Constants
    zero_nib = np.zeros(16)
    zero_nib[0] = 1.0
    ones_byte = np.zeros(256)
    ones_byte[0xFF] = 1.0
    carry_zero = np.zeros(2)
    carry_zero[0] = 1.0
    carry_one = np.zeros(2)
    carry_one[1] = 1.0

    for name, arr in [("zero_nib", zero_nib), ("ones_byte", ones_byte),
                      ("carry_zero", carry_zero), ("carry_one", carry_one)]:
        dims = list(arr.shape)
        flat = arr.flatten().astype(float)
        data = [float_to_fixed(v) for v in flat]
        sparse = tensor_to_sparse_coo(data)
        tensors.append((name, dims, data, sparse))

    return tensors


def convert_dir_to_c4onnx_v3(input_dir, output_path, verbose=False):
    """Convert a directory of .onnx files to a single .c4onnx v3 file."""
    onnx_files = sorted(glob.glob(os.path.join(input_dir, "*.onnx")))
    if not onnx_files:
        print(f"Error: no .onnx files found in {input_dir}")
        return False

    all_tensors = []  # (name, dims, fixed_data, sparse_info)
    all_subgraphs = []

    for onnx_path in onnx_files:
        if verbose:
            print(f"Processing {os.path.basename(onnx_path)}...")

        weight_tensors, subgraph = parse_onnx_for_v3(onnx_path, verbose)

        # Assign global tensor indices starting from current count
        base_idx = len(all_tensors)
        sg_name, num_inputs, num_outputs, num_temps, temp_sizes, nodes = subgraph

        # Add weight tensors with fixed-point conversion + sparse COO
        for wname, arr in weight_tensors:
            dims = list(arr.shape)
            flat = arr.flatten().astype(float)
            data = [float_to_fixed(v) for v in flat]
            sparse = tensor_to_sparse_coo(data)
            all_tensors.append((wname, dims, data, sparse))

            if verbose:
                nz = sum(1 for v in data if v != 0)
                tag = f"SPARSE {sparse[2]}" if sparse else "dense"
                print(f"    {wname}: {dims} ({nz} nz, {tag})")

        # Remap weight references in nodes: add base_idx offset
        # Note: OP_SCALE and OP_SLICE have literal values (not tensor refs)
        # in their second/third inputs — don't remap those.
        remapped_nodes = []
        for op, inputs, outputs in nodes:
            new_inputs = []
            for idx_i, inp in enumerate(inputs):
                is_tensor_ref = isinstance(inp, int) and inp >= 0
                # OP_SCALE: input[1] is literal scalar, not tensor ref
                if op == OP_SCALE_V3 and idx_i == 1:
                    is_tensor_ref = False
                # OP_SLICE: input[1] and input[2] are literal start/end
                if op == OP_SLICE_V3 and idx_i >= 1:
                    is_tensor_ref = False
                if is_tensor_ref:
                    new_inputs.append(inp + base_idx)
                else:
                    new_inputs.append(inp)
            remapped_nodes.append((op, new_inputs, outputs))

        all_subgraphs.append((sg_name, num_inputs, num_outputs, num_temps,
                              temp_sizes, remapped_nodes))

    # Add auxiliary tensors (shift tables, constants) needed by bundled runtime
    aux_tensors = build_auxiliary_tensors()
    all_tensors.extend(aux_tensors)
    if verbose:
        print(f"Added {len(aux_tensors)} auxiliary tensors (shift tables, constants)")

    # Write v3 binary
    with open(output_path, 'wb') as f:
        write_int(f, 0x584E4E4F)  # "ONNX" magic
        write_int(f, 3)            # Version 3
        write_int(f, len(all_tensors))
        write_int(f, len(all_subgraphs))

        # Tensors
        total_dense = 0
        total_sparse = 0
        for name, dims, data, sparse in all_tensors:
            name_bytes = name.encode('utf-8')[:63]
            write_int(f, len(name_bytes))
            f.write(name_bytes)

            write_int(f, len(dims))
            for d in dims:
                write_int(f, d)

            if sparse:
                indices, values, nnz = sparse
                write_int(f, 1)  # storage_type = sparse_coo
                write_int(f, nnz)
                for idx in indices:
                    write_int(f, idx)
                for val in values:
                    write_int(f, val)
                total_sparse += 1
            else:
                write_int(f, 0)  # storage_type = dense
                write_int(f, len(data))
                for val in data:
                    write_int(f, val)
                total_dense += 1

        # Subgraphs
        for sg_name, num_inputs, num_outputs, num_temps, temp_sizes, nodes in all_subgraphs:
            name_bytes = sg_name.encode('utf-8')[:63]
            write_int(f, len(name_bytes))
            f.write(name_bytes)

            write_int(f, num_inputs)
            write_int(f, num_outputs)
            write_int(f, num_temps)
            for ts in temp_sizes:
                write_int(f, ts)

            write_int(f, len(nodes))
            for op_type, inputs, outputs in nodes:
                write_int(f, op_type)
                write_int(f, len(inputs))
                for inp in inputs:
                    write_int(f, inp)
                write_int(f, len(outputs))
                for out in outputs:
                    write_int(f, out)

    file_size = os.path.getsize(output_path)
    print(f"Converted {len(onnx_files)} ONNX files -> {output_path}")
    print(f"  Tensors: {len(all_tensors)} ({total_sparse} sparse, {total_dense} dense)")
    print(f"  Subgraphs: {len(all_subgraphs)} ({', '.join(s[0] for s in all_subgraphs)})")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    return True


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to C4 binary format')
    parser.add_argument('input', help='Input ONNX file or directory')
    parser.add_argument('-o', '--output', help='Output C4 binary file (for --to-c4onnx-v3)')
    parser.add_argument('--to-c4onnx-v3', action='store_true',
                        help='Convert directory of ONNX files to c4onnx v3')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.to_c4onnx_v3:
        output = args.output or 'output.c4onnx'
        convert_dir_to_c4onnx_v3(args.input, output, args.verbose)
    else:
        # Legacy: single file conversion
        output = args.output or args.input.replace('.onnx', '.c4onnx')
        convert_onnx_to_c4(args.input, output, args.verbose)


if __name__ == '__main__':
    main()
