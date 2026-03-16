#!/usr/bin/env python3
"""
Export full C4TransformerVM weights to .c4onnx v3 binary format.

v3 adds:
  - Sparse COO storage for weight matrices (91%+ sparse)
  - Named subgraphs defining computation (b2n, n2b, nib_add, nib_mul, etc.)

The C runtime interprets the subgraphs instead of hardcoding the neural ops.

Usage:
    python export_vm_weights.py -o model.c4onnx
"""

import struct
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer_vm import C4TransformerVM, NeuralALU
import torch


SCALE = 65536  # 16.16 fixed-point

# Op types matching onnx_runtime_c4.c
OP_ADD_NN = 1
OP_MATMUL = 5
OP_SIGMOID_NN = 7
OP_SOFTMAX = 10
OP_CONCAT = 14
OP_SLICE = 15
OP_SCALE = 16


def float_to_fixed(val):
    """Convert float to 16.16 fixed-point int32."""
    result = int(val * SCALE)
    if result > 2147483647:
        result = 2147483647
    elif result < -2147483648:
        result = -2147483648
    return result


def write_int(f, val):
    """Write 32-bit little-endian integer."""
    if val < 0:
        val = val + 0x100000000
    f.write(struct.pack('<I', val & 0xFFFFFFFF))


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
    # Only use sparse if it saves space: nnz*2 < total_size
    if nnz * 2 < len(flat_data):
        return indices, values, nnz
    return None


def build_base_model(verbose=True):
    """Build the base VM model tensors and subgraphs.

    Returns (tensors, tensor_name_to_idx, subgraphs) where:
      tensors: list of (name, dims, flat_data, sparse_info)
      tensor_name_to_idx: dict mapping name -> index
      subgraphs: list of (name, num_inputs, num_outputs, num_temps, temp_sizes, nodes)
    """

    if verbose:
        print("Building C4TransformerVM...")

    vm = C4TransformerVM()
    alu = vm.alu

    # Collect all weight tensors from the NeuralALU
    tensors = []  # list of (name, dims, flat_data, sparse_info)
    tensor_name_to_idx = {}  # name -> index in tensors list

    def add_tensor(name, tensor):
        dims = list(tensor.shape)
        flat = tensor.flatten().tolist()
        data = [float_to_fixed(v) for v in flat]
        sparse = tensor_to_sparse_coo(data)
        idx = len(tensors)
        tensors.append((name, dims, data, sparse))
        tensor_name_to_idx[name] = idx
        if verbose:
            nonzero = sum(1 for v in flat if abs(v) > 1e-6)
            total = len(data)
            if sparse:
                print(f"  {name}: {dims} ({total} values, {nonzero} nz, SPARSE {sparse[2]} entries)")
            else:
                print(f"  {name}: {dims} ({total} values, {nonzero} nz, dense)")

    # Byte-to-nibble conversion
    add_tensor("b2n_W1", alu.b2n.W1)
    add_tensor("b2n_W2", alu.b2n.W2)

    # Nibble-to-byte conversion
    add_tensor("n2b_W1", alu.n2b.W1)
    add_tensor("n2b_W2", alu.n2b.W2)

    # Nibble add with carry
    add_tensor("nib_add_W1", alu.nib_add.W1)
    add_tensor("nib_add_W2_sum", alu.nib_add.W2_sum)
    add_tensor("nib_add_W2_cout", alu.nib_add.W2_cout)

    # Nibble multiply table
    add_tensor("nib_mul_W1", alu.nib_mul.W1)
    add_tensor("nib_mul_W2_lo", alu.nib_mul.W2_lo)
    add_tensor("nib_mul_W2_hi", alu.nib_mul.W2_hi)

    # Bitwise nibble tables
    add_tensor("nib_and_W1", alu.nib_and.W1)
    add_tensor("nib_and_W2", alu.nib_and.W2)
    add_tensor("nib_or_W1", alu.nib_or.W1)
    add_tensor("nib_or_W2", alu.nib_or.W2)
    add_tensor("nib_xor_W1", alu.nib_xor.W1)
    add_tensor("nib_xor_W2", alu.nib_xor.W2)

    # Shift tables
    add_tensor("shl_result", alu.shl_result)
    add_tensor("shl_overflow", alu.shl_overflow)
    add_tensor("shr_result", alu.shr_result)
    add_tensor("shr_underflow", alu.shr_underflow)

    # Constants
    add_tensor("zero_nib", alu.zero_nib)
    add_tensor("ones_byte", alu.ones_byte)
    add_tensor("carry_zero", alu.carry_zero)
    add_tensor("carry_one", alu.carry_one)

    # Detection matrices
    add_tensor("is_zero_W", alu.byte_is_zero_W)
    # Nibble-level negativity check: [16, 2], nibbles 8-15 are "negative"
    nib_is_neg_W = torch.zeros(16, 2)
    nib_is_neg_W[:8, 1] = 1.0   # 0-7 → not negative [0, 1]
    nib_is_neg_W[8:, 0] = 1.0   # 8-15 → negative [1, 0]
    add_tensor("is_neg_W", nib_is_neg_W)

    # ============ Build subgraphs ============
    TEMP_100 = 100 * SCALE  # temperature scalar in 16.16 fixed-point

    def T(name):
        return tensor_name_to_idx[name]

    subgraphs = []

    # --- b2n: byte[256] -> (hi[16], lo[16]) ---
    subgraphs.append(("b2n", 1, 2, 2, [256, 32], [
        (OP_MATMUL, [-1, T("b2n_W1")], [-2]),
        (OP_MATMUL, [-2, T("b2n_W2")], [-3]),
        (OP_SLICE,  [-3, 0, 16],       [-4]),
        (OP_SLICE,  [-3, 16, 32],      [-5]),
    ]))

    # --- n2b: (hi[16], lo[16]) -> byte[256] ---
    subgraphs.append(("n2b", 2, 1, 4, [32, 256, 256, 256], [
        (OP_CONCAT,  [-1, -2],              [-3]),
        (OP_MATMUL,  [-3, T("n2b_W1")],     [-4]),
        (OP_SCALE,   [-4, TEMP_100],         [-5]),
        (OP_SOFTMAX, [-5],                   [-6]),
        (OP_MATMUL,  [-6, T("n2b_W2")],      [-7]),
    ]))

    # --- nib_add: (a[16], b[16], cin[2]) -> (sum[16], cout[2]) ---
    subgraphs.append(("nib_add", 3, 2, 4, [34, 512, 512, 512], [
        (OP_CONCAT,  [-1, -2, -3],                [-4]),
        (OP_MATMUL,  [-4, T("nib_add_W1")],       [-5]),
        (OP_SCALE,   [-5, TEMP_100],               [-6]),
        (OP_SOFTMAX, [-6],                         [-7]),
        (OP_MATMUL,  [-7, T("nib_add_W2_sum")],   [-8]),
        (OP_MATMUL,  [-7, T("nib_add_W2_cout")],  [-9]),
    ]))

    # --- nib_and: (a[16], b[16]) -> result[16] ---
    subgraphs.append(("nib_and", 2, 1, 4, [32, 256, 256, 256], [
        (OP_CONCAT,  [-1, -2],              [-3]),
        (OP_MATMUL,  [-3, T("nib_and_W1")], [-4]),
        (OP_SCALE,   [-4, TEMP_100],         [-5]),
        (OP_SOFTMAX, [-5],                   [-6]),
        (OP_MATMUL,  [-6, T("nib_and_W2")], [-7]),
    ]))

    # --- nib_or: (a[16], b[16]) -> result[16] ---
    subgraphs.append(("nib_or", 2, 1, 4, [32, 256, 256, 256], [
        (OP_CONCAT,  [-1, -2],             [-3]),
        (OP_MATMUL,  [-3, T("nib_or_W1")], [-4]),
        (OP_SCALE,   [-4, TEMP_100],        [-5]),
        (OP_SOFTMAX, [-5],                  [-6]),
        (OP_MATMUL,  [-6, T("nib_or_W2")], [-7]),
    ]))

    # --- nib_xor: (a[16], b[16]) -> result[16] ---
    subgraphs.append(("nib_xor", 2, 1, 4, [32, 256, 256, 256], [
        (OP_CONCAT,  [-1, -2],              [-3]),
        (OP_MATMUL,  [-3, T("nib_xor_W1")], [-4]),
        (OP_SCALE,   [-4, TEMP_100],          [-5]),
        (OP_SOFTMAX, [-5],                    [-6]),
        (OP_MATMUL,  [-6, T("nib_xor_W2")],  [-7]),
    ]))

    # --- nib_mul: (a[16], b[16]) -> (lo[16], hi[16]) ---
    subgraphs.append(("nib_mul", 2, 2, 4, [32, 256, 256, 256], [
        (OP_CONCAT,  [-1, -2],                [-3]),
        (OP_MATMUL,  [-3, T("nib_mul_W1")],   [-4]),
        (OP_SCALE,   [-4, TEMP_100],            [-5]),
        (OP_SOFTMAX, [-5],                      [-6]),
        (OP_MATMUL,  [-6, T("nib_mul_W2_lo")], [-7]),
        (OP_MATMUL,  [-6, T("nib_mul_W2_hi")], [-8]),
    ]))

    # --- byte_is_zero: byte[256] -> result[2] ---
    subgraphs.append(("byte_is_zero", 1, 1, 0, [], [
        (OP_MATMUL, [-1, T("is_zero_W")], [-2]),
    ]))

    # --- nib_is_neg: nibble[16] -> result[2] ---
    subgraphs.append(("nib_is_neg", 1, 1, 0, [], [
        (OP_MATMUL, [-1, T("is_neg_W")], [-2]),
    ]))

    return tensors, tensor_name_to_idx, subgraphs


def write_c4onnx(path, tensors, subgraphs, verbose=True):
    """Write tensors and subgraphs to a .c4onnx v3 binary file."""
    num_subgraphs = len(subgraphs)

    with open(path, 'wb') as f:
        # Header
        write_int(f, 0x584E4E4F)  # "ONNX" magic
        write_int(f, 3)            # Version 3
        write_int(f, len(tensors))
        write_int(f, num_subgraphs)

        # Tensors (with sparse support)
        total_dense = 0
        total_sparse = 0
        for name, dims, data, sparse in tensors:
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
        for sg_name, num_inputs, num_outputs, num_temps, temp_sizes, nodes in subgraphs:
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

    file_size = os.path.getsize(path)

    if verbose:
        total_values = sum(len(d) for _, _, d, _ in tensors)
        sparse_saved = sum(
            (len(d) - s[2] * 2) * 4
            for _, _, d, s in tensors if s
        )
        print(f"\nExported {len(tensors)} tensors ({total_values:,} values)")
        print(f"  Dense: {total_dense}, Sparse: {total_sparse}")
        print(f"  Sparse savings: {sparse_saved:,} bytes ({sparse_saved/1024:.1f} KB)")
        print(f"Exported {num_subgraphs} subgraphs: {', '.join(s[0] for s in subgraphs)}")
        print(f"File: {path} ({file_size:,} bytes / {file_size/1024:.1f} KB)")

    return path


def export_vm_weights(output_path, verbose=True):
    """Export all C4TransformerVM weight tensors to .c4onnx v3 format."""
    tensors, tensor_name_to_idx, subgraphs = build_base_model(verbose=verbose)
    return write_c4onnx(output_path, tensors, subgraphs, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description='Export C4TransformerVM weights')
    parser.add_argument('-o', '--output', default='models/transformer_vm.c4onnx',
                        help='Output .c4onnx file')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    export_vm_weights(args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
