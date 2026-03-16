"""
Test chunk-generic MUL at BIT/PAIR/NIBBLE/BYTE.
"""

import random
import torch
import sys

from ..chunk_config import BIT, PAIR, NIBBLE, BYTE, ChunkConfig
from ..ops.mul import build_mul_layers, _compute_carry_passes
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


def make_embedding(a_chunks, b_chunks, config, opcode):
    ge = GenericE(config)
    N = config.num_positions
    x = torch.zeros(1, N, ge.DIM, dtype=config.torch_dtype)
    for i in range(N):
        x[0, i, ge.NIB_A] = float(a_chunks[i])
        x[0, i, ge.NIB_B] = float(b_chunks[i])
        x[0, i, ge.OP_START + opcode] = 1.0
    return x


def test_mul(config, opcode=27, num_cases=10000):
    max_val = (1 << config.total_bits) - 1
    layers = build_mul_layers(config, opcode)
    ge = GenericE(config)
    failures = 0

    random.seed(42)
    for _ in range(num_cases):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        expected = (a * b) & max_val

        a_chunks = int_to_chunks(a, config)
        b_chunks = int_to_chunks(b, config)
        x = make_embedding(a_chunks, b_chunks, config, opcode)

        with torch.no_grad():
            for layer in layers:
                x = layer(x)

        result_chunks = [round(x[0, i, ge.RESULT].item()) for i in range(config.num_positions)]
        result = chunks_to_int(result_chunks, config)

        if result != expected:
            failures += 1
            if failures <= 5:
                print(f"  FAIL MUL: {a:#010x} * {b:#010x} = {result:#010x} (expected {expected:#010x})")

    return failures


def test_mul_all():
    print("=" * 60)
    print("Chunk-Generic MUL Test")
    print("=" * 60)

    all_pass = True

    configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    for name, config in configs:
        passes = _compute_carry_passes(config)
        num_layers = 1 + len(passes) + 3  # schoolbook + carry passes + gen/look/final
        print(f"\n{name} (carry passes={len(passes)}, total layers={num_layers}):")
        f = test_mul(config)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {status}")
        if f > 0:
            all_pass = False

    print()
    if all_pass:
        print("All MUL tests passed!")
    else:
        print("Some MUL tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_mul_all()
    sys.exit(0 if success else 1)
