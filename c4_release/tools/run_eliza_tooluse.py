#!/usr/bin/env python3
"""
Run ELIZA with Tool-Use I/O.

Demonstrates how the VM pauses for I/O operations and generates
tool calls that are handled externally.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from tooluse_io import ToolUseVM, ToolUseIOHandler, ToolCallType


def run_eliza_with_tooluse(source: str, conversation: list[str], interactive: bool = False):
    """
    Run ELIZA program with tool-use I/O.

    Args:
        source: C source code
        conversation: List of input lines (ignored if interactive)
        interactive: If True, get input from user interactively
    """
    print("=" * 70)
    print("  ELIZA with TOOL-USE I/O")
    print("=" * 70)
    print()

    # Compile
    print("Compiling ELIZA...")
    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions, {len(data) if data else 0} bytes data")
    print()

    # Setup I/O handler
    output_buffer = []
    input_idx = [0]  # Use list for closure

    def output_callback(s):
        output_buffer.append(s)
        print(s, end='', flush=True)

    def input_callback():
        if interactive:
            return input() + '\n'
        else:
            if input_idx[0] < len(conversation):
                line = conversation[input_idx[0]]
                input_idx[0] += 1
                print(f"\033[36m{line}\033[0m")  # Cyan for user input
                return line + '\n'
            else:
                return ''  # EOF

    handler = ToolUseIOHandler(
        output_callback=output_callback,
        input_callback=input_callback
    )

    # Create VM
    vm = ToolUseVM(handler)
    vm.load(bytecode, data)

    # Run with step-by-step I/O handling
    print("-" * 70)
    print()

    start_time = time.time()
    steps = 0
    io_calls = 0
    max_steps = 10_000_000

    while steps < max_steps and not vm.halted:
        call = vm.step()

        if call:
            io_calls += 1
            # Handle the tool call
            response = handler.handle(call)
            vm.provide_response(response)

            # Show tool call details (for debugging)
            if call.call_type not in (ToolCallType.PUTCHAR, ToolCallType.EXIT):
                # Don't spam output with every putchar
                pass

        steps += 1

    exec_time = time.time() - start_time

    print()
    print("-" * 70)
    print()

    # Report
    print("=" * 70)
    print("  EXECUTION REPORT")
    print("=" * 70)
    print(f"Total steps:    {steps:,}")
    print(f"I/O tool calls: {io_calls:,}")
    print(f"Execution time: {exec_time*1000:.1f}ms")
    print(f"Exit code:      {vm.ax}")
    print()

    # Analyze tool call history
    call_types = {}
    for call in handler.call_history:
        call_types[call.call_type.value] = call_types.get(call.call_type.value, 0) + 1

    print("TOOL CALL BREAKDOWN:")
    for call_type, count in sorted(call_types.items()):
        print(f"  {call_type:15s}: {count:,}")

    print("=" * 70)

    return {
        'steps': steps,
        'io_calls': io_calls,
        'exec_time': exec_time,
        'output': ''.join(output_buffer),
        'call_history': handler.call_history
    }


def run_interactive_only():
    """Run Eliza in pure interactive mode with minimal output."""
    c_file = os.path.join(os.path.dirname(__file__), "eliza_simple.c")
    with open(c_file, 'r') as f:
        source = f.read()

    from src.compiler import compile_c
    bytecode, data = compile_c(source)

    handler = ToolUseIOHandler(
        output_callback=lambda s: print(s, end='', flush=True),
        input_callback=lambda: input() + '\n'
    )

    vm = ToolUseVM(handler)
    vm.load(bytecode, data)

    try:
        vm.run(max_steps=10_000_000)
    except (EOFError, KeyboardInterrupt):
        print("\n\nSession ended.")


def main():
    # Read ELIZA source
    c_file = os.path.join(os.path.dirname(__file__), "eliza_simple.c")
    with open(c_file, 'r') as f:
        source = f.read()

    # Check for interactive mode
    interactive = '--interactive' in sys.argv or '-i' in sys.argv

    # Pure interactive mode - just run the chat
    if interactive:
        run_interactive_only()
        return

    # Simulated conversation for demo/test
    conversation = [
        "hello",
        "i feel sad today",
        "i think about my mother",
        "i had a dream",
        "talking to a computer helps",
        "it makes me happy",
        "yes i am certain",
        "bye"
    ]

    result = run_eliza_with_tooluse(source, conversation, interactive=False)

    # Print final output summary
    print()
    print("FULL OUTPUT:")
    print("-" * 70)
    print(result['output'])


if __name__ == "__main__":
    main()
