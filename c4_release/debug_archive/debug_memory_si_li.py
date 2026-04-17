#!/usr/bin/env python3
"""Debug script to investigate SI/LI memory operation failures.

This script runs a minimal SI/LI test using the proper AutoregressiveVMRunner
and shows detailed results.
"""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token, _SetDim

def main():
    print("=" * 80)
    print("DEBUG: SI/LI Memory Operation Investigation (Using Runner)")
    print("=" * 80)

    # Create minimal test: write 42 to address 0x1000, read it back
    addr = 0x1000
    value = 42

    bytecode = [
        Opcode.IMM | (addr << 8),  # AX = 0x1000
        Opcode.PSH,                 # push AX (STACK0 = 0x1000)
        Opcode.IMM | (value << 8),  # AX = 42
        Opcode.SI,                  # memory[STACK0] = AX (store 42 at 0x1000)
        Opcode.IMM | (addr << 8),   # AX = 0x1000
        Opcode.LI,                  # AX = memory[AX] (load from 0x1000)
        Opcode.EXIT,
    ]

    print(f"\nBytecode: {len(bytecode)} instructions")
    print(f"  Write {value} to address 0x{addr:04x}")
    print(f"  Read back from address 0x{addr:04x}")
    print(f"  Expected exit code: {value}")

    # Run using proper runner
    print("\n" + "-" * 80)
    print("Running with AutoregressiveVMRunner...")

    runner = AutoregressiveVMRunner()
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        runner.model = runner.model.cuda()

    output, exit_code = runner.run(bytecode, max_steps=20)

    print(f"\nOutput length: {len(output)} bytes")
    print(f"Output (hex): {output.hex()}")
    print(f"Output (ascii, if printable): {repr(output)}")
    print(f"Exit code: {exit_code}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Expected exit code: {value}")
    print(f"Actual exit code:   {exit_code}")
    if exit_code == value:
        print("✓ PASS - Memory operations working!")
    else:
        print(f"✗ FAIL - Expected {value}, got {exit_code} (0x{exit_code:08x})")

        if exit_code == 0x01010101:
            print("\n⚠️  ERROR SIGNATURE: 0x01010101 (all bytes = 1)")
            print("   This indicates the model is generating byte_01 for all outputs.")
            print("   Possible causes:")
            print("   1. MEM sections not being generated (SI not working)")
            print("   2. MEM_STORE flags not being set (embedding issue)")
            print("   3. L15 memory lookup not finding values")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
