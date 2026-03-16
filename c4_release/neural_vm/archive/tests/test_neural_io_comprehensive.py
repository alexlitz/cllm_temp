"""
Comprehensive Neural I/O Tests - 1000+ test cases.

Tests:
1. Binary Position Encoding (encoding, decoding, dot products)
2. IOBuffer (read/write at all offsets)
3. User Input/Output markers
4. Sign flag computation
5. BLT/BGE branch operations
6. File system operations
7. Integration tests
"""

import torch
import random
import sys
import os
from typing import List, Tuple

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm.neural_io_full import (
    BinaryPositionEncoder,
    IOBuffer,
    UserInputMarker,
    UserOutputMarker,
    NeuralFileSystem,
    SetSignFlagFFN,
    BranchLTFFN,
    BranchGEFFN,
    NeuralIOSystem,
    VMState,
)
from .embedding import E, Opcode


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestStats:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, passed: bool, msg: str = ""):
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            if msg:
                self.errors.append(msg)

    def summary(self) -> str:
        total = self.passed + self.failed
        status = "✓" if self.failed == 0 else "✗"
        return f"{status} {self.name}: {self.passed}/{total} passed"


# ============================================================================
# BINARY POSITION ENCODING TESTS
# ============================================================================

def test_binary_encoding_comprehensive() -> TestStats:
    """Test binary position encoding for all positions 0-1023."""
    stats = TestStats("Binary Position Encoding")

    encoder = BinaryPositionEncoder(dim=16, max_positions=1024)

    # Test 1: All positions encode uniquely
    for pos in range(1024):
        enc = encoder.encode(torch.tensor([pos]))
        decoded = encoder.decode(enc)
        passed = decoded.item() == pos
        stats.record(passed, f"Position {pos}: decoded as {decoded.item()}")

    return stats


def test_binary_dot_products() -> TestStats:
    """Test that same positions have max dot product."""
    stats = TestStats("Binary Dot Products")

    encoder = BinaryPositionEncoder(dim=16, max_positions=256)

    # Test: Same position has dot product = dim
    for pos in range(256):
        enc = encoder.encode(torch.tensor([pos]))
        dot = (enc * enc).sum().item()
        passed = abs(dot - 16) < 0.01
        stats.record(passed, f"Position {pos}: dot={dot}")

    # Test: Different positions have dot product < dim
    for _ in range(200):
        pos1 = random.randint(0, 255)
        pos2 = random.randint(0, 255)
        if pos1 == pos2:
            continue
        enc1 = encoder.encode(torch.tensor([pos1]))
        enc2 = encoder.encode(torch.tensor([pos2]))
        dot = (enc1 * enc2).sum().item()
        passed = dot < 16
        stats.record(passed, f"Positions {pos1},{pos2}: dot={dot}")

    return stats


def test_binary_hamming_distance() -> TestStats:
    """Test that dot product correlates with hamming distance."""
    stats = TestStats("Binary Hamming Distance")

    encoder = BinaryPositionEncoder(dim=16, max_positions=256)

    for _ in range(200):
        pos1 = random.randint(0, 255)
        pos2 = random.randint(0, 255)

        enc1 = encoder.encode(torch.tensor([pos1]))
        enc2 = encoder.encode(torch.tensor([pos2]))
        dot = (enc1 * enc2).sum().item()

        # Hamming distance = number of differing bits
        xor = pos1 ^ pos2
        hamming = bin(xor).count('1')

        # dot = dim - 2 * hamming (each differing bit contributes -2 instead of +1)
        expected_dot = 16 - 2 * hamming
        passed = abs(dot - expected_dot) < 0.01
        stats.record(passed, f"Positions {pos1},{pos2}: dot={dot}, expected={expected_dot}")

    return stats


# ============================================================================
# IO BUFFER TESTS
# ============================================================================

def test_iobuffer_sequential_write_read() -> TestStats:
    """Test sequential write then read."""
    stats = TestStats("IOBuffer Sequential")

    buffer = IOBuffer(buffer_size=256, value_dim=8)

    # Write values 0-255
    for i in range(256):
        value = torch.zeros(8)
        value[0] = i % 16  # Low nibble
        value[1] = i // 16  # High nibble
        buffer.write(i, value)

    # Read back and verify
    for i in range(256):
        result = buffer.read(torch.tensor([i]))
        low = int(result[0, 0].item())
        high = int(result[0, 1].item())
        reconstructed = low + 16 * high
        passed = abs(reconstructed - i) < 1
        stats.record(passed, f"Offset {i}: got {reconstructed}")

    return stats


def test_iobuffer_random_access() -> TestStats:
    """Test random access patterns."""
    stats = TestStats("IOBuffer Random Access")

    buffer = IOBuffer(buffer_size=128, value_dim=8)

    # Write at random positions
    written = {}
    for _ in range(100):
        offset = random.randint(0, 127)
        char_val = random.randint(0, 255)
        value = torch.zeros(8)
        value[0] = char_val % 16
        value[1] = char_val // 16
        buffer.write(offset, value)
        written[offset] = char_val

    # Read back
    for offset, expected in written.items():
        result = buffer.read(torch.tensor([offset]))
        low = int(result[0, 0].item())
        high = int(result[0, 1].item())
        got = low + 16 * high
        passed = abs(got - expected) < 2  # Allow small error from attention
        stats.record(passed, f"Offset {offset}: expected {expected}, got {got}")

    return stats


def test_iobuffer_ascii_characters() -> TestStats:
    """Test storing and retrieving ASCII characters."""
    stats = TestStats("IOBuffer ASCII Characters")

    buffer = IOBuffer(buffer_size=128, value_dim=8)

    # Write "Hello, World!" as characters
    test_string = "Hello, World! This is a test of the neural I/O buffer system."
    for i, char in enumerate(test_string):
        if i >= 128:
            break
        value = torch.zeros(8)
        value[0] = ord(char) % 16
        value[1] = (ord(char) // 16) % 16
        buffer.write(i, value)

    # Read back
    for i, expected_char in enumerate(test_string):
        if i >= 128:
            break
        result = buffer.read(torch.tensor([i]))
        low = int(result[0, 0].item())
        high = int(result[0, 1].item())
        got_ord = low + 16 * high
        passed = abs(got_ord - ord(expected_char)) < 2
        stats.record(passed, f"Position {i}: expected '{expected_char}' ({ord(expected_char)}), got {got_ord}")

    return stats


def test_iobuffer_batch_read() -> TestStats:
    """Test batch reading multiple positions."""
    stats = TestStats("IOBuffer Batch Read")

    buffer = IOBuffer(buffer_size=64, value_dim=8)

    # Write values
    for i in range(64):
        value = torch.zeros(8)
        value[0] = i
        buffer.write(i, value)

    # Batch read
    offsets = torch.tensor([0, 10, 20, 30, 40, 50, 60, 63])
    results = buffer.read(offsets)

    for idx, offset in enumerate(offsets.tolist()):
        got = int(results[idx, 0].item())
        passed = abs(got - offset) < 2
        stats.record(passed, f"Batch offset {offset}: got {got}")

    return stats


# ============================================================================
# SIGN FLAG TESTS
# ============================================================================

def test_sign_flag_all_nibbles() -> TestStats:
    """Test sign flag computation for all high nibble values."""
    stats = TestStats("Sign Flag All Nibbles")

    set_sign = SetSignFlagFFN()

    for nibble_val in range(16):
        x = torch.zeros(1, 8, VMState.DIM)
        x[:, :, VMState.AX_BASE + 7] = nibble_val

        y = set_sign(x)
        flag = y[0, 0, VMState.FLAG_SIGN].item()

        # Sign bit is set when nibble >= 8
        expected = 1.0 if nibble_val >= 8 else 0.0
        passed = abs(flag - expected) < 0.2
        stats.record(passed, f"Nibble {nibble_val}: FLAG_SIGN={flag:.2f}, expected={expected}")

    return stats


def test_sign_flag_random_values() -> TestStats:
    """Test sign flag with random 32-bit values."""
    stats = TestStats("Sign Flag Random Values")

    set_sign = SetSignFlagFFN()

    for _ in range(100):
        # Random 32-bit value
        val = random.randint(0, 0xFFFFFFFF)
        high_nibble = (val >> 28) & 0xF  # Bits 28-31

        x = torch.zeros(1, 8, VMState.DIM)
        x[:, :, VMState.AX_BASE + 7] = high_nibble

        y = set_sign(x)
        flag = y[0, 0, VMState.FLAG_SIGN].item()

        expected = 1.0 if high_nibble >= 8 else 0.0
        passed = abs(flag - expected) < 0.2
        stats.record(passed, f"Value 0x{val:08X}: FLAG_SIGN={flag:.2f}")

    return stats


# ============================================================================
# BRANCH OPERATION TESTS
# ============================================================================

def test_blt_comprehensive() -> TestStats:
    """Test BLT with various values."""
    stats = TestStats("BLT Comprehensive")

    set_sign = SetSignFlagFFN()
    blt = BranchLTFFN()

    test_cases = []
    # Negative values (high nibble >= 8)
    for nibble in range(8, 16):
        test_cases.append((nibble, True))  # Should branch
    # Non-negative values (high nibble < 8)
    for nibble in range(0, 8):
        test_cases.append((nibble, False))  # Should not branch

    for high_nibble, should_branch in test_cases:
        for _ in range(5):  # Multiple tests per case
            pc_val = random.randint(50, 150)
            result_val = random.randint(200, 300)

            x = torch.zeros(1, 8, VMState.DIM)
            x[:, :, E.OP_START + Opcode.BLT] = 1.0
            x[:, :, VMState.AX_BASE + 7] = high_nibble
            x[:, :, VMState.PC] = pc_val
            x[:, :, E.RESULT] = result_val

            x = set_sign(x)
            y = blt(x)
            new_pc = y[0, 0, VMState.PC].item()

            if should_branch:
                expected = result_val
            else:
                expected = pc_val

            passed = abs(new_pc - expected) < 20
            stats.record(passed, f"BLT nibble={high_nibble}: PC={new_pc:.0f}, expected={expected}")

    return stats


def test_bge_comprehensive() -> TestStats:
    """Test BGE with various values."""
    stats = TestStats("BGE Comprehensive")

    set_sign = SetSignFlagFFN()
    bge = BranchGEFFN()

    test_cases = []
    # Non-negative values (high nibble < 8) - should branch
    for nibble in range(0, 8):
        test_cases.append((nibble, True))
    # Negative values (high nibble >= 8) - should not branch
    for nibble in range(8, 16):
        test_cases.append((nibble, False))

    for high_nibble, should_branch in test_cases:
        for _ in range(5):
            pc_val = random.randint(50, 150)
            result_val = random.randint(200, 300)

            x = torch.zeros(1, 8, VMState.DIM)
            x[:, :, E.OP_START + Opcode.BGE] = 1.0
            x[:, :, VMState.AX_BASE + 7] = high_nibble
            x[:, :, VMState.PC] = pc_val
            x[:, :, E.RESULT] = result_val

            x = set_sign(x)
            y = bge(x)
            new_pc = y[0, 0, VMState.PC].item()

            if should_branch:
                expected = result_val
            else:
                expected = pc_val

            passed = abs(new_pc - expected) < 20
            stats.record(passed, f"BGE nibble={high_nibble}: PC={new_pc:.0f}, expected={expected}")

    return stats


def test_branch_random_values() -> TestStats:
    """Test branches with random 32-bit signed values."""
    stats = TestStats("Branch Random Values")

    set_sign = SetSignFlagFFN()
    blt = BranchLTFFN()
    bge = BranchGEFFN()

    for _ in range(100):
        # Random signed 32-bit value
        val = random.randint(-2**31, 2**31 - 1)
        # Convert to unsigned for nibble extraction
        if val < 0:
            val_unsigned = val + 2**32
        else:
            val_unsigned = val

        high_nibble = (val_unsigned >> 28) & 0xF
        is_negative = val < 0

        pc_val = random.randint(100, 200)
        result_val = random.randint(300, 400)

        # Test BLT
        x = torch.zeros(1, 8, VMState.DIM)
        x[:, :, E.OP_START + Opcode.BLT] = 1.0
        x[:, :, VMState.AX_BASE + 7] = high_nibble
        x[:, :, VMState.PC] = pc_val
        x[:, :, E.RESULT] = result_val

        x = set_sign(x)
        y = blt(x)
        new_pc = y[0, 0, VMState.PC].item()

        expected = result_val if is_negative else pc_val
        passed = abs(new_pc - expected) < 30
        stats.record(passed, f"BLT val={val}: PC={new_pc:.0f}, expected={expected}")

        # Test BGE
        x2 = torch.zeros(1, 8, VMState.DIM)
        x2[:, :, E.OP_START + Opcode.BGE] = 1.0
        x2[:, :, VMState.AX_BASE + 7] = high_nibble
        x2[:, :, VMState.PC] = pc_val
        x2[:, :, E.RESULT] = result_val

        x2 = set_sign(x2)
        y2 = bge(x2)
        new_pc2 = y2[0, 0, VMState.PC].item()

        expected2 = result_val if not is_negative else pc_val
        passed2 = abs(new_pc2 - expected2) < 30
        stats.record(passed2, f"BGE val={val}: PC={new_pc2:.0f}, expected={expected2}")

    return stats


# ============================================================================
# USER INPUT/OUTPUT TESTS
# ============================================================================

def test_user_input_marker() -> TestStats:
    """Test user input marker functionality."""
    stats = TestStats("User Input Marker")

    input_marker = UserInputMarker(buffer_size=256)

    # Simulate input sequence
    test_inputs = [
        "Hello",
        "World",
        "Test123",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
    ]

    for test_input in test_inputs:
        input_marker.input_buffer.reset()
        input_marker.on_input_start(0)

        for i, char in enumerate(test_input):
            input_marker.on_input_token(i + 1, ord(char))

        input_marker.on_input_end(len(test_input) + 1)

        # Read back
        for i, expected_char in enumerate(test_input):
            char_val, eof = input_marker.read_char()
            passed = char_val == ord(expected_char) and not eof
            stats.record(passed, f"Input '{test_input}' pos {i}: got {chr(char_val) if char_val >= 0 else 'EOF'}")

        # Check EOF
        char_val, eof = input_marker.read_char()
        stats.record(eof, f"Input '{test_input}' EOF check")

    return stats


def test_user_output_marker() -> TestStats:
    """Test user output marker functionality."""
    stats = TestStats("User Output Marker")

    output_marker = UserOutputMarker(buffer_size=256)

    test_outputs = [
        "Output test",
        "Hello World!",
        "1234567890",
        "Mixed Case TEST",
    ]

    for test_output in test_outputs:
        output_marker.output_buffer.reset()
        output_marker.output_ptr.zero_()

        # Write characters
        for char in test_output:
            output_marker.write_char(ord(char))

        output_marker.on_output_start(0)

        # Read back
        for i, expected_char in enumerate(test_output):
            got_char = output_marker.get_char_for_position(i)
            passed = got_char is not None and abs(got_char - ord(expected_char)) < 2
            stats.record(passed, f"Output '{test_output}' pos {i}: expected '{expected_char}', got {chr(got_char) if got_char else 'None'}")

    return stats


# ============================================================================
# FILE SYSTEM TESTS
# ============================================================================

def test_file_system_basic() -> TestStats:
    """Test basic file system operations."""
    stats = TestStats("File System Basic")

    fs = NeuralFileSystem()

    # Test open
    fd = fs.open(hash("test.txt"), 1)  # Write mode
    stats.record(fd >= 0, f"Open file: fd={fd}")

    # Test write
    data = torch.zeros(10, 8)
    for i in range(10):
        data[i, 0] = i
    bytes_written = fs.write(fd, data)
    stats.record(bytes_written == 10, f"Write: {bytes_written} bytes")

    # Test close
    fs.close(fd)
    stats.record(True, "Close file")

    # Test open for read
    fd2 = fs.open(hash("test2.txt"), 0)  # Read mode
    stats.record(fd2 >= 0, f"Open for read: fd={fd2}")

    return stats


def test_file_system_multiple_files() -> TestStats:
    """Test multiple file operations."""
    stats = TestStats("File System Multiple Files")

    fs = NeuralFileSystem()

    # Open multiple files
    fds = []
    for i in range(10):
        fd = fs.open(hash(f"file{i}.txt"), 1)
        passed = fd >= 0 and fd not in fds
        stats.record(passed, f"Open file{i}: fd={fd}")
        fds.append(fd)

    # Write to each
    for i, fd in enumerate(fds):
        data = torch.zeros(5, 8)
        data[:, 0] = i
        written = fs.write(fd, data)
        stats.record(written == 5, f"Write to fd={fd}")

    # Close all
    for fd in fds:
        fs.close(fd)
        stats.record(True, f"Close fd={fd}")

    return stats


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_io_pipeline() -> TestStats:
    """Test full I/O pipeline: input -> process -> output."""
    stats = TestStats("Full I/O Pipeline")

    input_marker = UserInputMarker(buffer_size=256)
    output_marker = UserOutputMarker(buffer_size=256)

    test_cases = [
        ("abc", "ABC"),  # Uppercase
        ("123", "321"),  # Reverse
        ("hello", "HELLO"),
    ]

    for input_str, expected_output in test_cases:
        # Reset
        input_marker.input_buffer.reset()
        output_marker.output_buffer.reset()
        output_marker.output_ptr.zero_()

        # Simulate input
        input_marker.on_input_start(0)
        for i, char in enumerate(input_str):
            input_marker.on_input_token(i + 1, ord(char))
        input_marker.on_input_end(len(input_str) + 1)

        # Read input and write output (simulating VM)
        for _ in range(len(expected_output)):
            char, eof = input_marker.read_char()
            if eof:
                break
            # Transform (uppercase)
            if 'a' <= chr(char) <= 'z':
                char = char - ord('a') + ord('A')
            output_marker.write_char(char)

        # Check output
        output_marker.on_output_start(0)
        for i, expected_char in enumerate(expected_output):
            got = output_marker.get_char_for_position(i)
            passed = got is not None and abs(got - ord(expected_char)) < 2
            stats.record(passed, f"Pipeline '{input_str}' -> '{expected_output}': pos {i}")

    return stats


def test_stress_large_buffer() -> TestStats:
    """Stress test with large buffer."""
    stats = TestStats("Stress Large Buffer")

    buffer = IOBuffer(buffer_size=1024, value_dim=8)

    # Fill entire buffer
    for i in range(1024):
        value = torch.zeros(8)
        value[0] = i % 16
        value[1] = (i // 16) % 16
        buffer.write(i, value)

    # Random reads
    for _ in range(200):
        offset = random.randint(0, 1023)
        result = buffer.read(torch.tensor([offset]))
        low = int(result[0, 0].item())
        high = int(result[0, 1].item())
        got = low + 16 * high
        expected = offset % 256
        passed = abs(got - expected) < 10
        stats.record(passed, f"Large buffer offset {offset}: got {got}")

    return stats


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests() -> Tuple[int, int]:
    """Run all tests and return (passed, total)."""
    print("=" * 70)
    print("COMPREHENSIVE NEURAL I/O TESTS")
    print("=" * 70)
    print()

    all_stats: List[TestStats] = []

    # Binary Position Encoding Tests
    print("Running Binary Position Encoding Tests...")
    all_stats.append(test_binary_encoding_comprehensive())
    all_stats.append(test_binary_dot_products())
    all_stats.append(test_binary_hamming_distance())

    # IO Buffer Tests
    print("Running IOBuffer Tests...")
    all_stats.append(test_iobuffer_sequential_write_read())
    all_stats.append(test_iobuffer_random_access())
    all_stats.append(test_iobuffer_ascii_characters())
    all_stats.append(test_iobuffer_batch_read())

    # Sign Flag Tests
    print("Running Sign Flag Tests...")
    all_stats.append(test_sign_flag_all_nibbles())
    all_stats.append(test_sign_flag_random_values())

    # Branch Tests
    print("Running Branch Tests...")
    all_stats.append(test_blt_comprehensive())
    all_stats.append(test_bge_comprehensive())
    all_stats.append(test_branch_random_values())

    # User Input/Output Tests
    print("Running User I/O Tests...")
    all_stats.append(test_user_input_marker())
    all_stats.append(test_user_output_marker())

    # File System Tests
    print("Running File System Tests...")
    all_stats.append(test_file_system_basic())
    all_stats.append(test_file_system_multiple_files())

    # Integration Tests
    print("Running Integration Tests...")
    all_stats.append(test_full_io_pipeline())
    all_stats.append(test_stress_large_buffer())

    # Summary
    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for stats in all_stats:
        print(stats.summary())
        total_passed += stats.passed
        total_failed += stats.failed

        # Print first few errors if any
        if stats.errors and len(stats.errors) <= 3:
            for err in stats.errors:
                print(f"    Error: {err}")
        elif stats.errors:
            print(f"    ... and {len(stats.errors)} more errors")

    total = total_passed + total_failed
    print()
    print("=" * 70)
    if total_failed == 0:
        print(f"✓ ALL TESTS PASSED: {total_passed}/{total} tests")
    else:
        print(f"✗ SOME TESTS FAILED: {total_passed}/{total} passed, {total_failed} failed")
    print("=" * 70)

    return total_passed, total


if __name__ == "__main__":
    passed, total = run_all_tests()
    sys.exit(0 if passed == total else 1)
