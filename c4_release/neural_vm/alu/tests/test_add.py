"""
Test chunk-generic ADD at all 6 configurations.

For each config, creates embedding tensors with chunk-encoded operands,
runs the 3-layer ADD pipeline, and verifies RESULT chunks match expected.
"""

import random
import torch
import sys

from ..chunk_config import BIT, PAIR, NIBBLE, BYTE, HALFWORD, WORD, ALL_CONFIGS, ChunkConfig
from ..ops.add import build_add_layers
from ..ops.common import GenericE


def int_to_chunks(val: int, config: ChunkConfig):
    """Convert integer to list of chunk values (LSB first)."""
    chunks = []
    mask = config.chunk_max
    for _ in range(config.num_positions):
        chunks.append(val & mask)
        val >>= config.chunk_bits
    return chunks


def chunks_to_int(chunks, config: ChunkConfig):
    """Convert list of chunk values (LSB first) to integer."""
    val = 0
    for i, c in enumerate(chunks):
        val += int(c) << (config.chunk_bits * i)
    return val


def make_embedding(a_chunks, b_chunks, config: ChunkConfig, opcode: int):
    """Create embedding tensor [1, N, DIM] with operands and opcode."""
    ge = GenericE(config)
    N = config.num_positions
    x = torch.zeros(1, N, ge.DIM, dtype=config.torch_dtype)
    for i in range(N):
        x[0, i, ge.NIB_A] = float(a_chunks[i])
        x[0, i, ge.NIB_B] = float(b_chunks[i])
        x[0, i, ge.OP_START + opcode] = 1.0
    return x


def test_add_config(config: ChunkConfig, opcode: int = 25, num_cases: int = 10000):
    """Test ADD for a single config with random cases."""
    max_val = (1 << config.total_bits) - 1
    layers = build_add_layers(config, opcode)
    ge = GenericE(config)
    failures = 0

    random.seed(42)
    for case_i in range(num_cases):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        expected = (a + b) & max_val  # mod 2^32

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
                print(f"  FAIL: {a} + {b} = {result} (expected {expected})")
                print(f"    a_chunks={a_chunks}, b_chunks={b_chunks}")
                print(f"    result_chunks={result_chunks}")

    return failures


def test_add_all():
    """Test ADD at all 6 chunk configurations."""
    print("=" * 60)
    print("Chunk-Generic ADD Test")
    print("=" * 60)

    opcode = 25  # Opcode.ADD
    all_pass = True

    configs = [
        ("BIT (1-bit, 32 pos, fp16)", BIT),
        ("PAIR (2-bit, 16 pos, fp16)", PAIR),
        ("NIBBLE (4-bit, 8 pos, fp32)", NIBBLE),
        ("BYTE (8-bit, 4 pos, fp32)", BYTE),
        ("HALFWORD (16-bit, 2 pos, fp64)", HALFWORD),
        ("WORD (32-bit, 1 pos, fp64)", WORD),
    ]

    for name, config in configs:
        failures = test_add_config(config, opcode)
        status = "PASS" if failures == 0 else f"FAIL ({failures})"
        print(f"  {name}: {status}")
        if failures > 0:
            all_pass = False

    print()
    if all_pass:
        print("All ADD tests passed!")
    else:
        print("Some ADD tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_add_all()
    sys.exit(0 if success else 1)
