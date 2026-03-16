#!/usr/bin/env python3
"""
ARCHIVED: ONNX weight-baking for old per-opcode pipeline.

Moved from tools/bake_program.py. These functions bake program tables into
.c4onnx neural model weights (C4TransformerVM, ONNX subgraphs). They are
not used by the autoregressive pipeline.

Usage (historical):
    python bake_program.py base_model.c4onnx source.c -o baked_model.c4onnx
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_vm_weights = None

def _load_vm_weights():
    """Lazy import to avoid pulling in PyTorch when only patch_bytecode is needed."""
    global _vm_weights
    if _vm_weights is None:
        from tools.export_vm_weights import (
            build_base_model, write_c4onnx, tensor_to_sparse_coo,
            float_to_fixed, SCALE,
            OP_ADD_NN, OP_MATMUL, OP_SIGMOID_NN, OP_SCALE,
        )
        _vm_weights = {
            'build_base_model': build_base_model, 'write_c4onnx': write_c4onnx,
            'tensor_to_sparse_coo': tensor_to_sparse_coo,
            'float_to_fixed': float_to_fixed, 'SCALE': SCALE,
            'OP_ADD_NN': OP_ADD_NN, 'OP_MATMUL': OP_MATMUL,
            'OP_SIGMOID_NN': OP_SIGMOID_NN, 'OP_SCALE': OP_SCALE,
        }
    return _vm_weights


def bake_table(subgraph_name, tensor_prefix, entries, base_tensor_idx, verbose=True):
    """Bake an address->value lookup table into neural weights.

    The math: decompose address into one-hot nibbles [128], MATMUL with
    selection matrix to get per-entry scores, threshold + sigmoid to get
    a one-hot router, then MATMUL with value matrix to select output.

    Args:
        subgraph_name: name for the subgraph (e.g. "baked_fetch")
        tensor_prefix: prefix for tensor names (e.g. "bk_fetch")
        entries: list of (address: int, value: list[int]) pairs
                 where values are in 16.16 fixed-point
        base_tensor_idx: starting index for new tensors
        verbose: print info

    Returns:
        (tensors, subgraph)
    """
    w = _load_vm_weights()
    SCALE = w['SCALE']
    float_to_fixed = w['float_to_fixed']
    tensor_to_sparse_coo = w['tensor_to_sparse_coo']
    OP_ADD_NN = w['OP_ADD_NN']
    OP_MATMUL = w['OP_MATMUL']
    OP_SIGMOID_NN = w['OP_SIGMOID_NN']
    OP_SCALE = w['OP_SCALE']

    N = len(entries)
    value_size = len(entries[0][1])

    # select: [128, N] -- 1.0 at (nib_pos*16 + nib_val, entry_idx)
    select_data = [0] * (128 * N)
    for idx, (addr, _) in enumerate(entries):
        for nib_pos in range(8):
            nib_val = (addr >> (nib_pos * 4)) & 0xF
            select_data[(nib_pos * 16 + nib_val) * N + idx] = SCALE

    # bias: [N] -- all = -(8 - 0.5) * SCALE
    bias_val = float_to_fixed(-(8 - 0.5))
    bias_data = [bias_val] * N

    # val: [N, value_size]
    val_data = [0] * (N * value_size)
    for idx, (_, vals) in enumerate(entries):
        for j, v in enumerate(vals):
            val_data[idx * value_size + j] = v

    if verbose:
        sel_nnz = sum(1 for v in select_data if v != 0)
        val_nnz = sum(1 for v in val_data if v != 0)
        print(f"  {tensor_prefix}_select: [128, {N}] ({sel_nnz} nnz / {128*N} total)")
        print(f"  {tensor_prefix}_bias:   [{N}]")
        print(f"  {tensor_prefix}_val:    [{N}, {value_size}] ({val_nnz} nnz / {N*value_size} total)")

    tensors = [
        (f"{tensor_prefix}_select", [128, N], select_data,
         tensor_to_sparse_coo(select_data)),
        (f"{tensor_prefix}_bias", [N], bias_data,
         tensor_to_sparse_coo(bias_data)),
        (f"{tensor_prefix}_val", [N, value_size], val_data,
         tensor_to_sparse_coo(val_data)),
    ]

    t_select = base_tensor_idx
    t_bias = base_tensor_idx + 1
    t_val = base_tensor_idx + 2

    SCALE_100 = 100 * SCALE

    # input_0=-1, temp_0=-2..temp_3=-5, output_0=-6
    subgraph = (subgraph_name, 1, 1, 4, [N, N, N, N], [
        (OP_MATMUL,     [-1, t_select],  [-2]),   # scores = nibbles @ select
        (OP_ADD_NN,     [-2, t_bias],    [-3]),   # shifted = scores + bias
        (OP_SCALE,      [-3, SCALE_100], [-4]),   # scaled = shifted * 100
        (OP_SIGMOID_NN, [-4],            [-5]),   # router = sigmoid(scaled)
        (OP_MATMUL,     [-5, t_val],     [-6]),   # output = router @ val
    ])

    return tensors, subgraph


def nval_encode_fixed(val):
    """Encode an integer as NVal (4 one-hot bytes) in fixed-point."""
    SCALE = _load_vm_weights()['SCALE']
    result = [0] * 1024
    val_u32 = val & 0xFFFFFFFF
    for byte_idx in range(4):
        byte_val = (val_u32 >> (byte_idx * 8)) & 0xFF
        result[byte_idx * 256 + byte_val] = SCALE
    return result


def bake_program(base_tensors, base_tensor_name_to_idx, base_subgraphs,
                 bytecode, data=None, verbose=True):
    """Bake compiled program into model weights.

    Bakes the instruction table (and data segment if present) as neural
    lookup tables using the same address->value math.

    Returns (tensors, subgraphs) with baked additions.
    """
    float_to_fixed = _load_vm_weights()['float_to_fixed']

    tensors = list(base_tensors)
    subgraphs = list(base_subgraphs)

    # === Instruction table ===
    instructions = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        instructions.append((op, imm))

    if verbose:
        print(f"Baking {len(instructions)} instructions...")

    instr_entries = []
    for idx, (op, imm) in enumerate(instructions):
        addr = idx * 8
        # value = [op_fixed_point, ...imm_nval_1024...]  (size 1025)
        value = [float_to_fixed(float(op))] + nval_encode_fixed(imm)
        instr_entries.append((addr, value))

    instr_tensors, instr_sg = bake_table(
        "baked_fetch", "bk_fetch", instr_entries, len(tensors), verbose=verbose)
    tensors.extend(instr_tensors)
    subgraphs.append(instr_sg)

    # === Data table ===
    if data and len(data) > 0:
        if verbose:
            print(f"Baking {len(data)} data bytes...")

        data_entries = []
        for i, byte_val in enumerate(data):
            addr = 0x10000 + i
            value = [float_to_fixed(float(byte_val))]
            data_entries.append((addr, value))

        data_tensors, data_sg = bake_table(
            "baked_data", "bk_data", data_entries, len(tensors), verbose=verbose)
        tensors.extend(data_tensors)
        subgraphs.append(data_sg)

    return tensors, subgraphs


def main():
    from tools.bake_program import patch_bytecode

    parser = argparse.ArgumentParser(
        description='Bake a program into neural model weights (ONNX pipeline)')
    parser.add_argument('base_model', help='Base .c4onnx model (ignored, rebuilt from scratch)')
    parser.add_argument('source', help='C source file to compile and bake')
    parser.add_argument('-o', '--output', required=True,
                        help='Output .c4onnx file with baked program')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--argv', action='store_true',
                        help='Add argv/argc support (reads from stdin)')
    args = parser.parse_args()

    verbose = not args.quiet

    from src.compiler import compile_c
    w = _load_vm_weights()
    build_base_model = w['build_base_model']
    write_c4onnx = w['write_c4onnx']

    # Build base model (we rebuild rather than parsing the binary)
    tensors, tensor_name_to_idx, subgraphs = build_base_model(verbose=verbose)

    # Compile source
    with open(args.source) as f:
        source = f.read()
    bytecode, data = compile_c(source)
    if verbose:
        print(f"\nCompiled {args.source}: {len(bytecode)} instructions, {len(data)} data bytes")

    # Patch bytecode: replace MSET/MCMP with subroutines, optionally add argv
    bytecode = patch_bytecode(bytecode, argv=args.argv)
    if verbose:
        print(f"After patching: {len(bytecode)} instructions")

    # Bake program into model
    tensors, subgraphs = bake_program(
        tensors, tensor_name_to_idx, subgraphs, bytecode, data=data, verbose=verbose)

    # Write combined model
    if verbose:
        print()
    write_c4onnx(args.output, tensors, subgraphs, verbose=verbose)


if __name__ == "__main__":
    main()
