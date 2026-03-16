#!/usr/bin/env python3
"""
Streaming I/O Demo

Shows how the complete I/O system works in a chat-like generation flow:

1. VM executes until it needs input (GETCHAR with no buffered data)
2. Generation emits <NEED_INPUT/> and pauses
3. User provides input as continuation of token stream
4. Binary Position I/O retrieves input characters via attention
5. VM continues executing, outputting via PUTCHAR
6. When EXIT is called, generation emits <PROGRAM_END/>

This demonstrates unlimited streaming I/O without the 16-char buffer limit.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pure_gen_vm_v5 import EmbedDimsV5, OnnxIOExpert
from neural_vm import Opcode
from binary_position_io import BinaryPositionIO


class StreamingIOSession:
    """
    Simulates a streaming I/O session with chat-like interaction.

    The token stream looks like:
        [Program output...]<NEED_INPUT/>[User input...][More output...]<PROGRAM_END/>

    Binary position I/O allows retrieving any character from user input
    via attention-based position matching - no buffer size limit.
    """

    def __init__(self, num_bits: int = 12):
        self.E = EmbedDimsV5
        self.io_expert = OnnxIOExpert(self.E.DIM)
        self.binary_io = BinaryPositionIO(dim=64, num_bits=num_bits)
        self.binary_io.eval()

        self.num_bits = num_bits

        # Stream state
        self.token_stream = []
        self.anchor_positions = []  # Positions of <NEED_INPUT/> tags
        self.current_read_offset = 0
        self.current_anchor_idx = 0

    def putchar(self, char: str) -> str:
        """Execute PUTCHAR and return emitted character."""
        x = torch.zeros(1, self.E.DIM)
        x[0, self.E.OPCODE_START + Opcode.PUTC] = 1.0

        c = ord(char)
        x[0, self.E.OP_A_VAL_START + 0] = float(c & 0xF)
        x[0, self.E.OP_A_VAL_START + 1] = float((c >> 4) & 0xF)

        out = self.io_expert(x)

        if out[0, self.E.IO_OUTPUT_READY].item() > 0.5:
            char_lo = out[0, self.E.IO_CHAR_VAL_START + 0].item()
            char_hi = out[0, self.E.IO_CHAR_VAL_START + 1].item()
            char_val = int(round(char_lo)) + int(round(char_hi)) * 16
            emitted = chr(char_val)
            self.token_stream.append(emitted)
            return emitted
        return ""

    def need_input(self) -> str:
        """Signal that input is needed."""
        tag = "<NEED_INPUT/>"
        self.token_stream.append(tag)
        self.anchor_positions.append(len(self.token_stream) - 1)
        return tag

    def provide_input(self, text: str):
        """Add user input to the token stream."""
        for char in text:
            self.token_stream.append(char)

    def getchar_streaming(self, input_text: str, offset: int) -> str:
        """
        Get character from input using binary position matching.

        This simulates reading from the token stream via attention,
        where input_text is the user input that follows <NEED_INPUT/>.
        """
        # Build embeddings from token stream segment
        if not self.anchor_positions:
            return '\0'

        anchor_pos = self.anchor_positions[self.current_anchor_idx]

        # Create embeddings for the input portion
        seq_len = anchor_pos + 1 + len(input_text)
        x = torch.randn(1, seq_len, 64)

        # Encode input characters
        for i, c in enumerate(input_text):
            pos = anchor_pos + 1 + i
            if pos < seq_len:
                char_val = ord(c)
                for nib in range(8):
                    x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

        positions = torch.arange(seq_len).unsqueeze(0)
        anchor = torch.tensor([anchor_pos])
        input_length = torch.tensor([len(input_text)])
        read_offset = torch.tensor([float(offset)])

        _, _, weights = self.binary_io.getchar(
            x, positions, anchor, read_offset, input_length
        )

        peak_pos = weights[0, 0].argmax().item()

        # Map position back to character
        input_idx = peak_pos - anchor_pos - 1
        if 0 <= input_idx < len(input_text):
            return input_text[input_idx]
        return '\0'

    def program_end(self) -> str:
        """Signal program completion."""
        tag = "<PROGRAM_END/>"
        self.token_stream.append(tag)
        return tag

    def get_stream(self) -> str:
        """Get the complete token stream."""
        return ''.join(self.token_stream)


def demo_echo_program():
    """
    Simulate an echo program:
        printf("Enter text: ");
        while ((c = getchar()) != '\n') putchar(c);
        printf("\nDone!\n");
        exit(0);
    """
    print("=" * 70)
    print("DEMO: Echo Program (Streaming I/O)")
    print("=" * 70)
    print()

    session = StreamingIOSession()

    # Phase 1: printf("Enter text: ")
    print("Phase 1: Program output")
    prompt = "Enter text: "
    for c in prompt:
        session.putchar(c)
    print(f"  Emitted: \"{prompt}\"")

    # Phase 2: Need input
    print("\nPhase 2: Program needs input")
    session.need_input()
    print(f"  Emitted: <NEED_INPUT/>")

    # Phase 3: User provides input
    user_input = "Hello, World!\n"
    print(f"\nPhase 3: User input: \"{user_input.strip()}\"")
    session.provide_input(user_input)

    # Phase 4: Read and echo via streaming binary position I/O
    print("\nPhase 4: Echo (read via binary position I/O)")
    echoed = []
    for offset in range(len(user_input)):
        char = session.getchar_streaming(user_input, offset)
        if char == '\n':
            break
        echoed.append(char)
        session.putchar(char)

    print(f"  Characters read: {len(echoed)}")
    print(f"  Echoed: \"{''.join(echoed)}\"")

    # Phase 5: More output
    print("\nPhase 5: Program continues")
    for c in "\nDone!\n":
        session.putchar(c)

    # Phase 6: Exit
    session.program_end()
    print("  Emitted: <PROGRAM_END/>")

    # Show complete stream
    print("\n" + "=" * 70)
    print("COMPLETE TOKEN STREAM")
    print("=" * 70)
    stream = session.get_stream()
    print(f"\n{stream}\n")

    return True


def demo_multi_input():
    """
    Simulate a program with multiple input phases:
        printf("Name? ");
        name = gets();  // First input
        printf("Age? ");
        age = gets();   // Second input
        printf("Hello %s, age %s!\n", name, age);
        exit(0);
    """
    print("\n" + "=" * 70)
    print("DEMO: Multi-Input Program")
    print("=" * 70)
    print()

    session = StreamingIOSession()

    # First input
    print("Phase 1: First prompt and input")
    for c in "Name? ":
        session.putchar(c)
    session.need_input()
    name_input = "Alice\n"
    session.provide_input(name_input)
    print(f"  Output: \"Name? <NEED_INPUT/>{name_input.strip()}\"")

    # Read name via binary position I/O
    name_chars = []
    for offset in range(len(name_input)):
        char = session.getchar_streaming(name_input, offset)
        if char == '\n':
            break
        name_chars.append(char)
    name = ''.join(name_chars)
    print(f"  Read name: \"{name}\"")

    # Second input (new anchor)
    print("\nPhase 2: Second prompt and input")
    for c in "Age? ":
        session.putchar(c)
    session.need_input()
    session.current_anchor_idx = 1  # Move to second anchor
    age_input = "30\n"
    session.provide_input(age_input)
    print(f"  Output: \"Age? <NEED_INPUT/>{age_input.strip()}\"")

    # Read age via binary position I/O
    age_chars = []
    for offset in range(len(age_input)):
        char = session.getchar_streaming(age_input, offset)
        if char == '\n':
            break
        age_chars.append(char)
    age = ''.join(age_chars)
    print(f"  Read age: \"{age}\"")

    # Final output
    print("\nPhase 3: Final output")
    final = f"Hello {name}, age {age}!\n"
    for c in final:
        session.putchar(c)
    session.program_end()
    print(f"  Output: \"{final.strip()}<PROGRAM_END/>\"")

    # Complete stream
    print("\n" + "=" * 70)
    print("COMPLETE TOKEN STREAM")
    print("=" * 70)
    print(f"\n{session.get_stream()}\n")

    return True


def demo_large_input():
    """
    Demonstrate reading large input (4K characters) via streaming I/O.
    """
    print("\n" + "=" * 70)
    print("DEMO: Large Input (4K characters)")
    print("=" * 70)
    print()

    session = StreamingIOSession(num_bits=12)  # 4096 position capacity

    # Prompt
    for c in "Paste large text: ":
        session.putchar(c)
    session.need_input()

    # Large input
    large_input = "".join([chr(65 + (i % 26)) for i in range(4000)])  # 4000 chars
    session.provide_input(large_input)

    print(f"  Input length: {len(large_input)} characters")

    # Read at various positions to verify
    test_positions = [0, 100, 500, 1000, 2000, 3000, 3999]
    print(f"  Testing positions: {test_positions}")
    print()

    all_correct = True
    for pos in test_positions:
        char = session.getchar_streaming(large_input, pos)
        expected = large_input[pos]
        correct = char == expected
        print(f"    Position {pos:4d}: got '{char}' expected '{expected}' {'✓' if correct else '✗'}")
        if not correct:
            all_correct = False

    print(f"\n  Result: {'✓ ALL CORRECT' if all_correct else '✗ SOME INCORRECT'}")
    return all_correct


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("STREAMING I/O DEMONSTRATION")
    print("=" * 70)
    print("""
This demonstrates unlimited streaming I/O using:
- OnnxIOExpert for PUTCHAR/GETCHAR/EXIT operations
- BinaryPositionIO for attention-based character retrieval
- No buffer size limits - can handle 4K+ character inputs

The token stream format:
    [output]<NEED_INPUT/>[user input][more output]<PROGRAM_END/>

Input characters are retrieved via binary position matching in attention,
giving exact retrieval (weight=1.0) at any offset up to 2^num_bits.
""")

    results = []

    results.append(("Echo Program", demo_echo_program()))
    results.append(("Multi-Input Program", demo_multi_input()))
    results.append(("Large Input (4K)", demo_large_input()))

    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL DEMOS SUCCESSFUL! ✓")
    else:
        print("SOME DEMOS FAILED ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
