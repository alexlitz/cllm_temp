"""
Test chunk-generic SHL/SHR at BIT/PAIR/NIBBLE/BYTE.
"""

import random
import torch
import sys

from ..chunk_config import BIT, PAIR, NIBBLE, BYTE, ChunkConfig
from ..ops.shift import build_shl_layers, build_shr_layers
from ..ops.common import GenericE


def int_to_chunks(val, config):
    chunks = []
    mask = config.chunk_max
    for _ in range(config.num_positions):
        chunks.append(val & mask)
        val >>= config.chunk_bits
    return chunks


def chunks_to_int(chunks, config):
    val = 0
    for i, c in enumerate(chunks):
        val += int(c) << (config.chunk_bits * i)
    return val


def make_shift_embedding(a_chunks, shift_amount, config, opcode):
    """Encode operand A in NIB_A, shift amount in NIB_B (as chunked scalar)."""
    ge = GenericE(config)
    N = config.num_positions
    base = config.base
    x = torch.zeros(1, N, ge.DIM, dtype=config.torch_dtype)
    for i in range(N):
        x[0, i, ge.NIB_A] = float(a_chunks[i])
        x[0, i, ge.OP_START + opcode] = 1.0

    # Encode shift amount as little-endian chunks in NIB_B
    sa = shift_amount
    for i in range(N):
        x[0, i, ge.NIB_B] = float(sa % base)
        sa //= base
        if sa == 0:
            break

    return x


def test_shift_op(config, builder, op_func, op_name, opcode, num_cases=10000):
    max_val = (1 << config.total_bits) - 1
    layers = builder(config, opcode)
    ge = GenericE(config)
    failures = 0

    random.seed(42)
    for _ in range(num_cases):
        a = random.randint(0, max_val)
        shift_amount = random.randint(0, 31)
        expected = op_func(a, shift_amount) & max_val

        a_chunks = int_to_chunks(a, config)
        x = make_shift_embedding(a_chunks, shift_amount, config, opcode)

        with torch.no_grad():
            for layer in layers:
                x = layer(x)

        result_chunks = [round(x[0, i, ge.RESULT].item()) for i in range(config.num_positions)]
        result = chunks_to_int(result_chunks, config)

        if result != expected:
            failures += 1
            if failures <= 5:
                print(f"  FAIL {op_name}: {a:#010x} {op_name} {shift_amount} = {result:#010x} (expected {expected:#010x})")

    return failures


def test_shift_all():
    print("=" * 60)
    print("Chunk-Generic Shift Test")
    print("=" * 60)

    all_pass = True

    configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    ops = [
        ("SHL", 23, build_shl_layers, lambda a, k: (a << k) & 0xFFFFFFFF),
        ("SHR", 24, build_shr_layers, lambda a, k: a >> k),
    ]

    for op_name, opcode, builder, op_func in ops:
        print(f"\n{op_name}:")
        for name, config in configs:
            f = test_shift_op(config, builder, op_func, op_name, opcode)
            status = "PASS" if f == 0 else f"FAIL ({f})"
            print(f"  {name}: {status}")
            if f > 0:
                all_pass = False

    print()
    if all_pass:
        print("All shift tests passed!")
    else:
        print("Some shift tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_shift_all()
    sys.exit(0 if success else 1)
