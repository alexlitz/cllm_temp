#!/usr/bin/env python3
"""
Test suite for I/O operations.

Tests both streaming and tool-use modes for:
- GETCHAR (character input)
- PUTCHAR (character output)
- EXIT (program termination)
"""

import torch
import sys
import os

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm import E, Opcode, SparseMoEALU, IOToolCallType
from neural_vm import IOHandler, StreamingIOHandler, ToolUseIOHandler
from neural_vm import ToolCall, ToolResponse


def encode_io_op(opcode: int, char_val: int = 0) -> torch.Tensor:
    """Encode an I/O operation."""
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    for i in range(E.NUM_POSITIONS):
        x[0, i, E.OP_START + opcode] = 1.0
        x[0, i, E.POS] = float(i)
    # Put char value in NIB_A at position 0 (for PUTCHAR)
    x[0, 0, E.NIB_A] = float(char_val & 0xFF)
    return x


def test_putchar_streaming():
    """Test PUTCHAR in streaming mode."""
    print("\n=== PUTCHAR Streaming Mode ===")
    passed = 0
    total = 0

    # Create embedding
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = StreamingIOHandler()

    # Test outputting "Hi"
    for c in "Hi":
        x[0, 0, E.IO_CHAR] = float(ord(c))
        x[0, 0, E.IO_OUTPUT_READY] = 1.0
        output = handler.check_io(x)

        total += 1
        if output == c:
            passed += 1
            print(f"  OK: PUTCHAR('{c}') -> '{output}'")
        else:
            print(f"  FAIL: PUTCHAR('{c}') -> '{output}', expected '{c}'")

    # Check buffer
    total += 1
    if handler.output_buffer == "Hi":
        passed += 1
        print(f"  OK: Buffer = '{handler.output_buffer}'")
    else:
        print(f"  FAIL: Buffer = '{handler.output_buffer}', expected 'Hi'")

    print(f"PUTCHAR Streaming: {passed}/{total} passed")
    return passed, total


def test_putchar_tooluse():
    """Test PUTCHAR in tool-use mode."""
    print("\n=== PUTCHAR Tool-Use Mode ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = ToolUseIOHandler()

    # Test PUTCHAR
    x[0, 0, E.IO_CHAR] = float(ord('X'))
    x[0, 0, E.IO_OUTPUT_READY] = 1.0
    output = handler.check_io(x)

    total += 1
    expected = 'TOOL_CALL:putchar:1:{"char": 88}'
    if output == expected:
        passed += 1
        print(f"  OK: {output}")
    else:
        print(f"  FAIL: Got '{output}'")
        print(f"        Expected '{expected}'")

    # Check tool call object
    total += 1
    call = handler.get_pending_call()
    if call and call.call_type == "putchar" and call.params.get("char") == 88:
        passed += 1
        print(f"  OK: ToolCall type={call.call_type}, char={call.params['char']}")
    else:
        print(f"  FAIL: Invalid tool call")

    print(f"PUTCHAR Tool-Use: {passed}/{total} passed")
    return passed, total


def test_getchar_streaming():
    """Test GETCHAR in streaming mode."""
    print("\n=== GETCHAR Streaming Mode ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = StreamingIOHandler()

    # Request input
    x[0, 0, E.IO_NEED_INPUT] = 1.0
    output = handler.check_io(x)

    total += 1
    if output == "<NEED_INPUT/>":
        passed += 1
        print(f"  OK: GETCHAR -> '{output}'")
    else:
        print(f"  FAIL: GETCHAR -> '{output}', expected '<NEED_INPUT/>'")

    total += 1
    if handler.waiting_for_input:
        passed += 1
        print(f"  OK: waiting_for_input = True")
    else:
        print(f"  FAIL: waiting_for_input = False")

    # Provide input
    handler.provide_input(ord('A'), x)

    total += 1
    if x[0, 0, E.IO_INPUT_READY].item() == 1.0:
        passed += 1
        print(f"  OK: IO_INPUT_READY = 1.0")
    else:
        print(f"  FAIL: IO_INPUT_READY = {x[0, 0, E.IO_INPUT_READY].item()}")

    total += 1
    char_val = int(x[0, 0, E.IO_CHAR].item())
    if char_val == ord('A'):
        passed += 1
        print(f"  OK: IO_CHAR = {char_val} ('A')")
    else:
        print(f"  FAIL: IO_CHAR = {char_val}, expected {ord('A')}")

    print(f"GETCHAR Streaming: {passed}/{total} passed")
    return passed, total


def test_getchar_tooluse():
    """Test GETCHAR in tool-use mode."""
    print("\n=== GETCHAR Tool-Use Mode ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = ToolUseIOHandler()

    # Request input
    x[0, 0, E.IO_NEED_INPUT] = 1.0
    output = handler.check_io(x)

    total += 1
    expected = 'TOOL_CALL:getchar:1:{}'
    if output == expected:
        passed += 1
        print(f"  OK: {output}")
    else:
        print(f"  FAIL: Got '{output}'")
        print(f"        Expected '{expected}'")

    # Handle response
    response = ToolResponse(call_id=1, success=True, result="B")
    handler.handle_tool_response(response, x)

    total += 1
    char_val = int(x[0, 0, E.IO_CHAR].item())
    if char_val == ord('B'):
        passed += 1
        print(f"  OK: Response handled, char = {char_val} ('B')")
    else:
        print(f"  FAIL: char = {char_val}, expected {ord('B')}")

    print(f"GETCHAR Tool-Use: {passed}/{total} passed")
    return passed, total


def test_exit_streaming():
    """Test EXIT in streaming mode."""
    print("\n=== EXIT Streaming Mode ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = StreamingIOHandler()

    # Trigger EXIT with code 42
    x[0, 0, E.IO_EXIT_CODE] = 42.0
    x[0, 0, E.IO_PROGRAM_END] = 1.0
    output = handler.check_io(x)

    total += 1
    expected = '<PROGRAM_END code="42"/>'
    if output == expected:
        passed += 1
        print(f"  OK: {output}")
    else:
        print(f"  FAIL: Got '{output}'")
        print(f"        Expected '{expected}'")

    total += 1
    if handler.finished:
        passed += 1
        print(f"  OK: finished = True")
    else:
        print(f"  FAIL: finished = False")

    total += 1
    if handler.exit_code == 42:
        passed += 1
        print(f"  OK: exit_code = 42")
    else:
        print(f"  FAIL: exit_code = {handler.exit_code}")

    print(f"EXIT Streaming: {passed}/{total} passed")
    return passed, total


def test_exit_tooluse():
    """Test EXIT in tool-use mode."""
    print("\n=== EXIT Tool-Use Mode ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = ToolUseIOHandler()

    # Trigger EXIT
    x[0, 0, E.IO_EXIT_CODE] = 0.0
    x[0, 0, E.IO_PROGRAM_END] = 1.0
    output = handler.check_io(x)

    total += 1
    expected = 'TOOL_CALL:exit:1:{"code": 0}'
    if output == expected:
        passed += 1
        print(f"  OK: {output}")
    else:
        print(f"  FAIL: Got '{output}'")
        print(f"        Expected '{expected}'")

    print(f"EXIT Tool-Use: {passed}/{total} passed")
    return passed, total


def test_input_buffer():
    """Test input string buffering."""
    print("\n=== Input Buffer ===")
    passed = 0
    total = 0

    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler = StreamingIOHandler()

    # Buffer "ABC"
    handler.provide_input_string("ABC", x)

    # Check first char
    total += 1
    char_val = int(x[0, 0, E.IO_CHAR].item())
    if char_val == ord('A'):
        passed += 1
        print(f"  OK: First char = 'A'")
    else:
        print(f"  FAIL: First char = {chr(char_val)}")

    # Get next char
    handler.next_input_char(x)
    total += 1
    char_val = int(x[0, 0, E.IO_CHAR].item())
    if char_val == ord('B'):
        passed += 1
        print(f"  OK: Second char = 'B'")
    else:
        print(f"  FAIL: Second char = {chr(char_val)}")

    # Get next char
    handler.next_input_char(x)
    total += 1
    char_val = int(x[0, 0, E.IO_CHAR].item())
    if char_val == ord('C'):
        passed += 1
        print(f"  OK: Third char = 'C'")
    else:
        print(f"  FAIL: Third char = {chr(char_val)}")

    # Try to get beyond buffer
    total += 1
    has_more = handler.next_input_char(x)
    if not has_more:
        passed += 1
        print(f"  OK: Buffer exhausted")
    else:
        print(f"  FAIL: Expected buffer to be exhausted")

    print(f"Input Buffer: {passed}/{total} passed")
    return passed, total


def test_tool_call_parsing():
    """Test ToolCall and ToolResponse parsing."""
    print("\n=== Tool Call Parsing ===")
    passed = 0
    total = 0

    # Test ToolCall.to_string and from_string
    call = ToolCall(call_type="putchar", call_id=42, params={"char": 72})
    s = call.to_string()
    total += 1
    expected = 'TOOL_CALL:putchar:42:{"char": 72}'
    if s == expected:
        passed += 1
        print(f"  OK: to_string = '{s}'")
    else:
        print(f"  FAIL: to_string = '{s}'")
        print(f"        Expected '{expected}'")

    # Parse back
    parsed = ToolCall.from_string(s)
    total += 1
    if parsed and parsed.call_type == "putchar" and parsed.call_id == 42:
        passed += 1
        print(f"  OK: from_string works")
    else:
        print(f"  FAIL: from_string failed")

    # Test ToolResponse
    response = ToolResponse(call_id=42, success=True, result={"status": "ok"})
    s = response.to_string()
    total += 1
    if "TOOL_RESPONSE:42:" in s:
        passed += 1
        print(f"  OK: Response to_string = '{s}'")
    else:
        print(f"  FAIL: Response to_string = '{s}'")

    print(f"Tool Call Parsing: {passed}/{total} passed")
    return passed, total


def test_alu_io_integration():
    """Test I/O FFNs in the ALU."""
    print("\n=== ALU I/O Integration ===")
    passed = 0
    total = 0

    try:
        alu = SparseMoEALU()
        print("ALU loaded successfully")

        # Test PUTCHAR opcode triggers I/O slots
        x = encode_io_op(Opcode.PUTCHAR, ord('Z'))
        y = alu(x)

        total += 1
        if y[0, 0, E.IO_OUTPUT_READY].item() > 0.5:
            passed += 1
            print(f"  OK: PUTCHAR sets IO_OUTPUT_READY")
        else:
            print(f"  FAIL: PUTCHAR didn't set IO_OUTPUT_READY")

        # Test EXIT opcode
        x = encode_io_op(Opcode.EXIT, 1)
        y = alu(x)

        total += 1
        if y[0, 0, E.IO_PROGRAM_END].item() > 0.5:
            passed += 1
            print(f"  OK: EXIT sets IO_PROGRAM_END")
        else:
            print(f"  FAIL: EXIT didn't set IO_PROGRAM_END")

        # Test GETCHAR opcode
        x = encode_io_op(Opcode.GETCHAR)
        y = alu(x)

        total += 1
        if y[0, 0, E.IO_NEED_INPUT].item() > 0.5:
            passed += 1
            print(f"  OK: GETCHAR sets IO_NEED_INPUT")
        else:
            print(f"  FAIL: GETCHAR didn't set IO_NEED_INPUT")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"ALU I/O Integration: {passed}/{total} passed")
    return passed, total


def main():
    print("=" * 60)
    print("I/O Operations Test Suite")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    p, t = test_putchar_streaming()
    total_passed += p
    total_tests += t

    p, t = test_putchar_tooluse()
    total_passed += p
    total_tests += t

    p, t = test_getchar_streaming()
    total_passed += p
    total_tests += t

    p, t = test_getchar_tooluse()
    total_passed += p
    total_tests += t

    p, t = test_exit_streaming()
    total_passed += p
    total_tests += t

    p, t = test_exit_tooluse()
    total_passed += p
    total_tests += t

    p, t = test_input_buffer()
    total_passed += p
    total_tests += t

    p, t = test_tool_call_parsing()
    total_passed += p
    total_tests += t

    p, t = test_alu_io_integration()
    total_passed += p
    total_tests += t

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)

    if total_passed == total_tests:
        print("\nAll I/O tests passed!")
    else:
        print(f"\n{total_tests - total_passed} tests failed")


if __name__ == "__main__":
    main()
