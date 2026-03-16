"""
Test chunk-generic MOD at NIBBLE/BYTE.
"""

import random
import torch
import sys

from ..chunk_config import NIBBLE, BYTE, ChunkConfig
from ..ops.mod import build_mod_layers
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


def test_mod(config, opcode=29, num_cases=10000):
    max_val = (1 << config.total_bits) - 1
    layers = build_mod_layers(config, opcode)
    ge = GenericE(config)
    failures = 0

    random.seed(42)
    for _ in range(num_cases):
        a = random.randint(0, max_val)
        b = random.randint(1, max_val)  # divisor >= 1
        expected = a % b

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
                print(f"  FAIL MOD: {a} % {b} = {result} (expected {expected})")

    return failures


def test_mod_all():
    print("=" * 60)
    print("Chunk-Generic MOD Test")
    print("=" * 60)

    all_pass = True

    configs = [
        ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        f = test_mod(config)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {status}")
        if f > 0:
            all_pass = False

    print()
    if all_pass:
        print("All MOD tests passed!")
    else:
        print("Some MOD tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_mod_all()
    sys.exit(0 if success else 1)
