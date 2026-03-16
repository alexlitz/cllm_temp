#!/usr/bin/env python3
"""
Full I/O System Test

Tests both:
1. Buffer-based I/O (existing OnnxIOExpert with soft-indexing)
2. Streaming I/O (new BinaryPositionIO for unlimited input)

The full I/O flow:
- PUTCHAR: Write char to IO_CHAR, set IO_OUTPUT_READY
- GETCHAR: Read char from buffer or stream via position matching
- EXIT: Set IO_PROGRAM_END
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pure_gen_vm_v5 import EmbedDimsV5, OnnxIOExpert
from neural_vm import Opcode
from binary_position_io import BinaryPositionIO, PowerOfTwoBitExtractor


def test_putchar():
    """Test PUTCHAR operation."""
    print("=" * 70)
    print("TEST: PUTCHAR")
    print("=" * 70)

    E = EmbedDimsV5
    io_expert = OnnxIOExpert(E.DIM)

    # Create embedding with PUTCHAR opcode and character 'H' (72)
    x = torch.zeros(1, E.DIM)

    # Set PUTCHAR opcode
    x[0, E.OPCODE_START + Opcode.PUTC] = 1.0

    # Set character 'H' (72 = 0x48) in OP_A
    # 72 = 4*16 + 8 → nibble 0 = 8, nibble 1 = 4
    x[0, E.OP_A_VAL_START + 0] = 8.0   # Low nibble
    x[0, E.OP_A_VAL_START + 1] = 4.0   # High nibble

    # Run IO expert
    out = io_expert(x)

    # Check results
    output_ready = out[0, E.IO_OUTPUT_READY].item()
    char_lo = out[0, E.IO_CHAR_VAL_START + 0].item()
    char_hi = out[0, E.IO_CHAR_VAL_START + 1].item()
    char_val = int(char_lo) + int(char_hi) * 16

    print(f"  Input: PUTCHAR('H') [72 = 0x48]")
    print(f"  IO_OUTPUT_READY: {output_ready:.1f} (expected: 1.0)")
    print(f"  IO_CHAR: nibbles [{char_lo:.0f}, {char_hi:.0f}] = {char_val} = '{chr(char_val)}'")

    passed = output_ready > 0.5 and char_val == 72
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_getchar_buffered():
    """Test GETCHAR with pre-loaded buffer."""
    print("\n" + "=" * 70)
    print("TEST: GETCHAR (Buffered)")
    print("=" * 70)

    E = EmbedDimsV5
    io_expert = OnnxIOExpert(E.DIM)

    # Create embedding with pre-loaded buffer "ABC"
    x = torch.zeros(1, E.DIM)

    # Load "ABC" into buffer
    # A=65 (0x41), B=66 (0x42), C=67 (0x43)
    for i, char in enumerate("ABC"):
        c = ord(char)
        x[0, E.IO_BUFFER_START + i*4 + 0] = float(c & 0xF)        # Low nibble
        x[0, E.IO_BUFFER_START + i*4 + 1] = float((c >> 4) & 0xF)  # High nibble

    # Set buffer length and position
    x[0, E.IO_BUFFER_LEN] = 3.0
    x[0, E.IO_BUFFER_POS] = 0.0

    # Set GETCHAR opcode
    x[0, E.OPCODE_START + Opcode.GETC] = 1.0

    all_passed = True

    for expected_char in "ABC":
        # Run IO expert
        out = io_expert(x)

        # Check result
        result_lo = out[0, E.RESULT_VAL_START + 0].item()
        result_hi = out[0, E.RESULT_VAL_START + 1].item()
        result_val = int(round(result_lo)) + int(round(result_hi)) * 16
        new_pos = out[0, E.IO_BUFFER_POS].item()
        need_input = out[0, E.IO_NEED_INPUT].item()

        expected_val = ord(expected_char)
        passed = result_val == expected_val and need_input < 0.5

        print(f"  GETCHAR at pos {int(x[0, E.IO_BUFFER_POS].item())}: "
              f"got {result_val}='{chr(result_val) if 32<=result_val<127 else '?'}' "
              f"expected {expected_val}='{expected_char}' "
              f"{'✓' if passed else '✗'}")

        if not passed:
            all_passed = False

        # Update state for next iteration
        x = out.clone()
        x[0, E.OPCODE_START:E.OPCODE_END] = 0.0
        x[0, E.OPCODE_START + Opcode.GETC] = 1.0

    # One more read should trigger NEED_INPUT
    out = io_expert(x)
    need_input = out[0, E.IO_NEED_INPUT].item()
    print(f"  GETCHAR at end: IO_NEED_INPUT={need_input:.1f} (expected: 1.0)")

    if need_input < 0.5:
        all_passed = False

    print(f"  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_getchar_streaming():
    """Test GETCHAR with streaming input via IO_INPUT_READY."""
    print("\n" + "=" * 70)
    print("TEST: GETCHAR (Streaming via IO_INPUT_READY)")
    print("=" * 70)

    E = EmbedDimsV5
    io_expert = OnnxIOExpert(E.DIM)

    all_passed = True

    for char in "Hi!":
        # Create embedding with streaming input
        x = torch.zeros(1, E.DIM)

        # Set IO_INPUT_READY and character in IO_CHAR
        x[0, E.IO_INPUT_READY] = 1.0
        c = ord(char)
        x[0, E.IO_CHAR_VAL_START + 0] = float(c & 0xF)
        x[0, E.IO_CHAR_VAL_START + 1] = float((c >> 4) & 0xF)

        # Set GETCHAR opcode
        x[0, E.OPCODE_START + Opcode.GETC] = 1.0

        # Run IO expert
        out = io_expert(x)

        # Check result
        result_lo = out[0, E.RESULT_VAL_START + 0].item()
        result_hi = out[0, E.RESULT_VAL_START + 1].item()
        result_val = int(round(result_lo)) + int(round(result_hi)) * 16

        # IO_INPUT_READY should be cleared after reading
        input_ready_after = out[0, E.IO_INPUT_READY].item()

        expected_val = ord(char)
        passed = result_val == expected_val and input_ready_after < 0.5

        print(f"  Streaming '{char}': got {result_val}='{chr(result_val) if 32<=result_val<127 else '?'}' "
              f"INPUT_READY after={input_ready_after:.1f} "
              f"{'✓' if passed else '✗'}")

        if not passed:
            all_passed = False

    print(f"  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_exit():
    """Test EXIT operation."""
    print("\n" + "=" * 70)
    print("TEST: EXIT")
    print("=" * 70)

    E = EmbedDimsV5
    io_expert = OnnxIOExpert(E.DIM)

    # Create embedding with EXIT opcode and exit code 42
    x = torch.zeros(1, E.DIM)

    # Set EXIT opcode
    x[0, E.OPCODE_START + Opcode.EXIT] = 1.0

    # Set exit code 42 (0x2A) in OP_A
    x[0, E.OP_A_VAL_START + 0] = 10.0  # Low nibble (A=10)
    x[0, E.OP_A_VAL_START + 1] = 2.0   # High nibble

    # Run IO expert
    out = io_expert(x)

    # Check results
    program_end = out[0, E.IO_PROGRAM_END].item()
    result_lo = out[0, E.RESULT_VAL_START + 0].item()
    result_hi = out[0, E.RESULT_VAL_START + 1].item()
    result_val = int(round(result_lo)) + int(round(result_hi)) * 16

    print(f"  Input: EXIT(42)")
    print(f"  IO_PROGRAM_END: {program_end:.1f} (expected: 1.0)")
    print(f"  Result: {result_val} (expected: 42)")

    passed = program_end > 0.5 and result_val == 42
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_binary_position_io():
    """Test binary position-based I/O for streaming input."""
    print("\n" + "=" * 70)
    print("TEST: Binary Position I/O (Attention-Based Streaming)")
    print("=" * 70)

    # Setup
    dim = 64
    num_bits = 12  # 4K positions
    batch = 1

    io_system = BinaryPositionIO(dim=dim, num_bits=num_bits)
    io_system.eval()

    # Simulate conversation:
    # "Hello<NEED_INPUT/>Alice says hi"
    #  0123456789...
    #        ^anchor=5 (position of <NEED_INPUT/>)

    input_text = "Alice says hi"
    seq_len = 6 + len(input_text)  # "Hello" + anchor + input
    anchor_pos = 5

    # Create embeddings
    x = torch.randn(batch, seq_len, dim)

    # Encode input characters at positions after anchor
    for i, c in enumerate(input_text):
        pos = anchor_pos + 1 + i
        char_val = ord(c)
        # Encode in first 8 dims
        for nib in range(8):
            x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    anchor = torch.tensor([anchor_pos])
    input_length = torch.tensor([len(input_text)])

    print(f"  Sequence: Hello<NI>{input_text}")
    print(f"  Anchor position: {anchor_pos}")
    print(f"  Input length: {len(input_text)}")
    print()

    # Read all characters via GETCHAR
    print("  GETCHAR sequence:")
    read_offset = torch.tensor([0.0])
    all_passed = True

    for i, expected_char in enumerate(input_text):
        _, new_offset, weights = io_system.getchar(
            x, positions, anchor, read_offset, input_length
        )

        peak_pos = weights[0, 0].argmax().item()
        peak_weight = weights[0, 0, peak_pos].item()
        expected_pos = anchor_pos + 1 + i

        passed = peak_pos == expected_pos and peak_weight > 0.99

        print(f"    offset={int(read_offset.item()):2d}: "
              f"peak@pos{peak_pos:2d} (w={peak_weight:.3f}) → '{expected_char}' "
              f"{'✓' if passed else '✗'}")

        if not passed:
            all_passed = False

        read_offset = new_offset

    print(f"\n  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_printf_simulation():
    """Simulate printf("Hello, ") followed by GETCHAR and echo."""
    print("\n" + "=" * 70)
    print("TEST: Printf + GETCHAR Simulation")
    print("=" * 70)

    E = EmbedDimsV5
    io_expert = OnnxIOExpert(E.DIM)

    output_stream = []

    # printf("Hello, ")
    print("  Phase 1: printf(\"Hello, \")")
    for char in "Hello, ":
        x = torch.zeros(1, E.DIM)
        x[0, E.OPCODE_START + Opcode.PUTC] = 1.0
        c = ord(char)
        x[0, E.OP_A_VAL_START + 0] = float(c & 0xF)
        x[0, E.OP_A_VAL_START + 1] = float((c >> 4) & 0xF)

        out = io_expert(x)

        if out[0, E.IO_OUTPUT_READY].item() > 0.5:
            char_lo = out[0, E.IO_CHAR_VAL_START + 0].item()
            char_hi = out[0, E.IO_CHAR_VAL_START + 1].item()
            char_val = int(round(char_lo)) + int(round(char_hi)) * 16
            output_stream.append(chr(char_val))

    print(f"    Output: {''.join(output_stream)}")

    # Simulate NEED_INPUT
    print("  Phase 2: <NEED_INPUT/> (waiting for input)")
    output_stream.append("<NEED_INPUT/>")

    # Simulate user input "World"
    user_input = "World"
    print(f"  Phase 3: User input: \"{user_input}\"")

    # GETCHAR loop to read user input (streaming mode)
    read_chars = []
    for char in user_input:
        x = torch.zeros(1, E.DIM)
        x[0, E.OPCODE_START + Opcode.GETC] = 1.0
        x[0, E.IO_INPUT_READY] = 1.0
        c = ord(char)
        x[0, E.IO_CHAR_VAL_START + 0] = float(c & 0xF)
        x[0, E.IO_CHAR_VAL_START + 1] = float((c >> 4) & 0xF)

        out = io_expert(x)

        result_lo = out[0, E.RESULT_VAL_START + 0].item()
        result_hi = out[0, E.RESULT_VAL_START + 1].item()
        result_val = int(round(result_lo)) + int(round(result_hi)) * 16
        read_chars.append(chr(result_val))

    print(f"    Read: {''.join(read_chars)}")

    # Echo user input
    print("  Phase 4: Echo (printf user input)")
    for char in read_chars:
        x = torch.zeros(1, E.DIM)
        x[0, E.OPCODE_START + Opcode.PUTC] = 1.0
        c = ord(char)
        x[0, E.OP_A_VAL_START + 0] = float(c & 0xF)
        x[0, E.OP_A_VAL_START + 1] = float((c >> 4) & 0xF)

        out = io_expert(x)

        if out[0, E.IO_OUTPUT_READY].item() > 0.5:
            char_lo = out[0, E.IO_CHAR_VAL_START + 0].item()
            char_hi = out[0, E.IO_CHAR_VAL_START + 1].item()
            char_val = int(round(char_lo)) + int(round(char_hi)) * 16
            output_stream.append(chr(char_val))

    # EXIT
    x = torch.zeros(1, E.DIM)
    x[0, E.OPCODE_START + Opcode.EXIT] = 1.0
    out = io_expert(x)

    if out[0, E.IO_PROGRAM_END].item() > 0.5:
        output_stream.append("<PROGRAM_END/>")

    print(f"\n  Full token stream: {''.join(output_stream)}")

    expected = "Hello, <NEED_INPUT/>World<PROGRAM_END/>"
    passed = ''.join(output_stream) == expected
    print(f"  Expected:          {expected}")
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_large_input_binary_io():
    """Test binary position I/O with larger inputs (4K characters)."""
    print("\n" + "=" * 70)
    print("TEST: Large Input (4K characters via Binary Position I/O)")
    print("=" * 70)

    dim = 64
    num_bits = 12  # 4096 positions
    batch = 1

    io_system = BinaryPositionIO(dim=dim, num_bits=num_bits)
    io_system.eval()

    # Create 4000 character input
    input_len = 4000
    input_text = "".join([chr(65 + (i % 26)) for i in range(input_len)])  # A-Z repeating

    anchor_pos = 100
    seq_len = anchor_pos + 1 + input_len

    x = torch.randn(batch, seq_len, dim)

    # Encode input
    for i, c in enumerate(input_text):
        pos = anchor_pos + 1 + i
        char_val = ord(c)
        for nib in range(8):
            x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    anchor = torch.tensor([anchor_pos])
    input_length = torch.tensor([input_len])

    # Test reading at various offsets
    test_offsets = [0, 1, 100, 500, 1000, 2000, 3000, 3999]
    all_passed = True

    print(f"  Input length: {input_len} characters")
    print(f"  Testing offsets: {test_offsets}")
    print()

    for offset in test_offsets:
        read_offset = torch.tensor([float(offset)])
        _, _, weights = io_system.getchar(x, positions, anchor, read_offset, input_length)

        peak_pos = weights[0, 0].argmax().item()
        peak_weight = weights[0, 0, peak_pos].item()
        expected_pos = anchor_pos + 1 + offset
        expected_char = input_text[offset]

        passed = peak_pos == expected_pos and peak_weight > 0.99

        print(f"    offset={offset:4d}: peak@pos{peak_pos:4d} (w={peak_weight:.4f}) "
              f"→ '{expected_char}' {'✓' if passed else '✗'}")

        if not passed:
            all_passed = False

    print(f"\n  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def main():
    """Run all I/O tests."""
    print("\n" + "=" * 70)
    print("FULL I/O SYSTEM TEST SUITE")
    print("=" * 70)

    tests = [
        ("PUTCHAR", test_putchar),
        ("GETCHAR (Buffered)", test_getchar_buffered),
        ("GETCHAR (Streaming)", test_getchar_streaming),
        ("EXIT", test_exit),
        ("Binary Position I/O", test_binary_position_io),
        ("Printf + GETCHAR Simulation", test_printf_simulation),
        ("Large Input (4K)", test_large_input_binary_io),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
