"""
Test chunk-generic CMP (LT/GT/LE/GE) at all 6 configurations.
"""

import random
import torch
import sys

from ..chunk_config import BIT, PAIR, NIBBLE, BYTE, HALFWORD, WORD, ChunkConfig
from ..ops.cmp import build_lt_layers, build_gt_layers, build_le_layers, build_ge_layers
from ..ops.common import GenericE


def int_to_chunks(val, config):
    chunks = []
    mask = config.chunk_max
    for _ in range(config.num_positions):
        chunks.append(val & mask)
        val >>= config.chunk_bits
    return chunks


def make_embedding(a_chunks, b_chunks, config, opcode):
    ge = GenericE(config)
    N = config.num_positions
    x = torch.zeros(1, N, ge.DIM, dtype=config.torch_dtype)
    for i in range(N):
        x[0, i, ge.NIB_A] = float(a_chunks[i])
        x[0, i, ge.NIB_B] = float(b_chunks[i])
        x[0, i, ge.OP_START + opcode] = 1.0
    return x


def test_cmp_op(config, builder, op_func, op_name, opcode, num_cases=10000):
    max_val = (1 << config.total_bits) - 1
    layers = builder(config, opcode)
    ge = GenericE(config)
    failures = 0

    random.seed(42)
    for _ in range(num_cases):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        expected = int(op_func(a, b))

        a_chunks = int_to_chunks(a, config)
        b_chunks = int_to_chunks(b, config)
        x = make_embedding(a_chunks, b_chunks, config, opcode)

        with torch.no_grad():
            for layer in layers:
                x = layer(x)

        result = round(x[0, 0, ge.RESULT].item())

        if result != expected:
            failures += 1
            if failures <= 5:
                print(f"  FAIL {op_name}: {a} vs {b} = {result} (expected {expected})")

    return failures


def test_cmp_all():
    print("=" * 60)
    print("Chunk-Generic CMP Test")
    print("=" * 60)

    all_pass = True

    configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE),
        ("BYTE", BYTE), ("HALFWORD", HALFWORD), ("WORD", WORD),
    ]

    ops = [
        ("LT", 19, build_lt_layers, lambda a, b: a < b),
        ("GT", 20, build_gt_layers, lambda a, b: a > b),
        ("LE", 21, build_le_layers, lambda a, b: a <= b),
        ("GE", 22, build_ge_layers, lambda a, b: a >= b),
    ]

    for op_name, opcode, builder, op_func in ops:
        print(f"\n{op_name}:")
        for name, config in configs:
            f = test_cmp_op(config, builder, op_func, op_name, opcode)
            status = "PASS" if f == 0 else f"FAIL ({f})"
            print(f"  {name}: {status}")
            if f > 0:
                all_pass = False

    print()
    if all_pass:
        print("All CMP tests passed!")
    else:
        print("Some CMP tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_cmp_all()
    sys.exit(0 if success else 1)
