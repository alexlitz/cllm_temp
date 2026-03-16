"""
Precision boundary tests for chunk-generic ALU.

Verifies that each config works at its precision boundaries:
- Edge cases: 0, 1, max, max-1
- Carry/borrow propagation chains (all P's active)
- Overflow wrapping
"""

import torch
import sys

from ..chunk_config import BIT, PAIR, NIBBLE, BYTE, HALFWORD, WORD, ChunkConfig
from ..ops.add import build_add_layers
from ..ops.sub import build_sub_layers
from ..ops.div import build_div_layers
from ..ops.mul import build_mul_layers
from ..ops.cmp import build_lt_layers, build_gt_layers, build_le_layers, build_ge_layers
from ..ops.bitwise import build_and_layers, build_or_layers, build_xor_layers
from ..ops.shift import build_shl_layers, build_shr_layers
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


def make_shift_embedding(a_chunks, shift_amount, config, opcode):
    ge = GenericE(config)
    N = config.num_positions
    base = config.base
    x = torch.zeros(1, N, ge.DIM, dtype=config.torch_dtype)
    for i in range(N):
        x[0, i, ge.NIB_A] = float(a_chunks[i])
        x[0, i, ge.OP_START + opcode] = 1.0
    sa = shift_amount
    for i in range(N):
        x[0, i, ge.NIB_B] = float(sa % base)
        sa //= base
        if sa == 0:
            break
    return x


def run_op(a, b, config, layers, opcode):
    ge = GenericE(config)
    a_chunks = int_to_chunks(a, config)
    b_chunks = int_to_chunks(b, config)
    x = make_embedding(a_chunks, b_chunks, config, opcode)
    with torch.no_grad():
        for layer in layers:
            x = layer(x)
    result_chunks = [round(x[0, i, ge.RESULT].item()) for i in range(config.num_positions)]
    return chunks_to_int(result_chunks, config)


def run_shift_op(a, shift, config, layers, opcode):
    ge = GenericE(config)
    a_chunks = int_to_chunks(a, config)
    x = make_shift_embedding(a_chunks, shift, config, opcode)
    with torch.no_grad():
        for layer in layers:
            x = layer(x)
    result_chunks = [round(x[0, i, ge.RESULT].item()) for i in range(config.num_positions)]
    return chunks_to_int(result_chunks, config)


def test_add_edge_cases(config, name):
    """Test ADD edge cases for a config."""
    max_val = (1 << config.total_bits) - 1
    layers = build_add_layers(config, 25)
    failures = 0

    cases = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (max_val, 0, max_val),
        (0, max_val, max_val),
        (max_val, 1, 0),                    # overflow wrap
        (1, max_val, 0),                    # overflow wrap
        (max_val, max_val, max_val - 1),    # double overflow
    ]

    # Full carry chain: all chunks at max (e.g., 0xFFFFFFFF + 1 = 0)
    cases.append((max_val, 1, 0))

    # Carry propagation: chunk_max at all positions, +1 at LSB
    if config.num_positions > 1:
        carry_chain = sum(config.chunk_max << (config.chunk_bits * i)
                          for i in range(config.num_positions - 1))
        cases.append((carry_chain, 1, carry_chain + 1))

    for a, b, expected in cases:
        result = run_op(a, b, config, layers, 25)
        if result != expected:
            failures += 1
            print(f"    FAIL {name}: {a} + {b} = {result} (expected {expected})")

    return failures


def test_sub_edge_cases(config, name):
    """Test SUB edge cases."""
    max_val = (1 << config.total_bits) - 1
    layers = build_sub_layers(config, 26)
    failures = 0

    cases = [
        (0, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, max_val),                    # underflow wrap
        (max_val, max_val, 0),
        (0, max_val, 1),                    # underflow wrap
    ]

    # Full borrow chain
    if config.num_positions > 1:
        val = 1 << config.chunk_bits  # e.g. 16 for NIBBLE
        cases.append((val, 1, val - 1))

    for a, b, expected in cases:
        result = run_op(a, b, config, layers, 26)
        if result != expected:
            failures += 1
            print(f"    FAIL {name}: {a} - {b} = {result} (expected {expected})")

    return failures


def test_div_edge_cases(config, name):
    """Test DIV edge cases."""
    max_val = (1 << config.total_bits) - 1
    layers = build_div_layers(config, 28)
    failures = 0

    cases = [
        (0, 1, 0),
        (1, 1, 1),
        (10, 3, 3),
        (100, 7, 14),
    ]

    # max_val tests only for fp64 configs (fp32 can't represent 2^32-1 exactly)
    if config.precision == "fp64":
        cases += [
            (max_val, 1, max_val),
            (max_val, max_val, 1),
            (max_val, 2, max_val // 2),
        ]
    else:
        cases += [
            (167000, 1, 167000),
            (167000, 7, 23857),
            (100000, 13, 7692),
        ]

    for a, b, expected in cases:
        result = run_op(a, b, config, layers, 28)
        if result != expected:
            failures += 1
            print(f"    FAIL {name}: {a} / {b} = {result} (expected {expected})")

    return failures


def test_mul_edge_cases(config, name):
    """Test MUL edge cases."""
    max_val = (1 << config.total_bits) - 1
    layers = build_mul_layers(config, 27)
    failures = 0

    cases = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
        (2, 3, 6),
        (max_val, 0, 0),
        (0, max_val, 0),
        (max_val, 1, max_val),
        (1, max_val, max_val),
        (max_val, 2, (max_val * 2) & max_val),     # overflow
        (2, max_val, (max_val * 2) & max_val),     # overflow
        (max_val, max_val, (max_val * max_val) & max_val),
    ]

    for a, b, expected in cases:
        result = run_op(a, b, config, layers, 27)
        if result != expected:
            failures += 1
            print(f"    FAIL {name}: {a} * {b} = {result} (expected {expected})")

    return failures


def test_cmp_edge_cases(config, name):
    """Test CMP (LT/GT/LE/GE) edge cases."""
    max_val = (1 << config.total_bits) - 1
    failures = 0

    lt_layers = build_lt_layers(config, 19)
    gt_layers = build_gt_layers(config, 20)
    le_layers = build_le_layers(config, 21)
    ge_layers = build_ge_layers(config, 22)

    cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (max_val, 0),
        (0, max_val),
        (max_val, max_val),
        (max_val - 1, max_val),
        (max_val, max_val - 1),
    ]

    for a, b in cases:
        for op_name, layers, opcode, func in [
            ("LT", lt_layers, 19, lambda x, y: int(x < y)),
            ("GT", gt_layers, 20, lambda x, y: int(x > y)),
            ("LE", le_layers, 21, lambda x, y: int(x <= y)),
            ("GE", ge_layers, 22, lambda x, y: int(x >= y)),
        ]:
            expected = func(a, b)
            result = run_op(a, b, config, layers, opcode)
            if result != expected:
                failures += 1
                print(f"    FAIL {name} {op_name}: {a} vs {b} = {result} (expected {expected})")

    return failures


def test_bitwise_edge_cases(config, name):
    """Test AND/OR/XOR edge cases."""
    max_val = (1 << config.total_bits) - 1
    failures = 0

    and_layers = build_and_layers(config, 16)
    or_layers = build_or_layers(config, 14)
    xor_layers = build_xor_layers(config, 15)

    cases = [
        (0, 0),
        (0, max_val),
        (max_val, 0),
        (max_val, max_val),
        (1, 1),
        (0xAA & max_val, 0x55 & max_val),  # alternating bits
    ]

    for a, b in cases:
        for op_name, layers, opcode, func in [
            ("AND", and_layers, 16, lambda x, y: x & y),
            ("OR", or_layers, 14, lambda x, y: x | y),
            ("XOR", xor_layers, 15, lambda x, y: x ^ y),
        ]:
            expected = func(a, b)
            result = run_op(a, b, config, layers, opcode)
            if result != expected:
                failures += 1
                print(f"    FAIL {name} {op_name}: {a:#x} op {b:#x} = {result:#x} (expected {expected:#x})")

    return failures


def test_shift_edge_cases(config, name):
    """Test SHL/SHR edge cases."""
    max_val = (1 << config.total_bits) - 1
    failures = 0

    shl_layers = build_shl_layers(config, 23)
    shr_layers = build_shr_layers(config, 24)

    cases = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (max_val, 0), (max_val, 1), (max_val, 31),
        (1, 31),
    ]

    for a, shift in cases:
        # SHL
        expected = (a << shift) & max_val
        result = run_shift_op(a, shift, config, shl_layers, 23)
        if result != expected:
            failures += 1
            print(f"    FAIL {name} SHL: {a:#x} << {shift} = {result:#x} (expected {expected:#x})")
        # SHR
        expected = a >> shift
        result = run_shift_op(a, shift, config, shr_layers, 24)
        if result != expected:
            failures += 1
            print(f"    FAIL {name} SHR: {a:#x} >> {shift} = {result:#x} (expected {expected:#x})")

    return failures


def test_mod_edge_cases(config, name):
    """Test MOD edge cases."""
    max_val = (1 << config.total_bits) - 1
    layers = build_mod_layers(config, 29)
    failures = 0

    cases = [
        (0, 1, 0),
        (1, 1, 0),
        (1, 2, 1),
        (10, 3, 1),
        (100, 7, 2),
        (7, 7, 0),
        (13, 7, 6),
        (15, 16, 15),
        (16, 16, 0),
        (17, 16, 1),
        (255, 16, 15),
        (256, 255, 1),
    ]

    for a, b, expected in cases:
        result = run_op(a, b, config, layers, 29)
        if result != expected:
            failures += 1
            print(f"    FAIL {name}: {a} % {b} = {result} (expected {expected})")

    return failures


def test_precision_all():
    print("=" * 60)
    print("Precision Boundary Tests")
    print("=" * 60)

    all_pass = True

    # ADD/SUB: all 6 configs
    add_sub_configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE),
        ("BYTE", BYTE), ("HALFWORD", HALFWORD), ("WORD", WORD),
    ]

    print("\nADD edge cases:")
    for name, config in add_sub_configs:
        f = test_add_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    print("\nSUB edge cases:")
    for name, config in add_sub_configs:
        f = test_sub_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # DIV: NIBBLE through WORD
    div_configs = [
        ("NIBBLE", NIBBLE), ("BYTE", BYTE),
        ("HALFWORD", HALFWORD), ("WORD", WORD),
    ]

    print("\nDIV edge cases:")
    for name, config in div_configs:
        f = test_div_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # MUL: BIT through BYTE
    mul_configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    print("\nMUL edge cases:")
    for name, config in mul_configs:
        f = test_mul_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # CMP: all 6 configs
    print("\nCMP edge cases:")
    for name, config in add_sub_configs:
        f = test_cmp_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # Bitwise: BIT through BYTE
    bitwise_configs = [
        ("BIT", BIT), ("PAIR", PAIR), ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    print("\nBitwise edge cases:")
    for name, config in bitwise_configs:
        f = test_bitwise_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # Shift: BIT through BYTE
    print("\nShift edge cases:")
    for name, config in bitwise_configs:
        f = test_shift_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    # MOD: NIBBLE, BYTE
    mod_configs = [
        ("NIBBLE", NIBBLE), ("BYTE", BYTE),
    ]

    print("\nMOD edge cases:")
    for name, config in mod_configs:
        f = test_mod_edge_cases(config, name)
        status = "PASS" if f == 0 else f"FAIL ({f})"
        print(f"  {name}: {status}")
        if f > 0:
            all_pass = False

    print()
    if all_pass:
        print("All precision tests passed!")
    else:
        print("Some precision tests FAILED!")
    return all_pass


if __name__ == "__main__":
    success = test_precision_all()
    sys.exit(0 if success else 1)
