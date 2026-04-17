#!/usr/bin/env python3
"""Test neural JMP with real compiled C code."""

import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

def test():
    print("=== Neural JMP with Compiled C Code ===\n")

    # Simple program that doesn't use control flow (baseline)
    runner1 = AutoregressiveVMRunner()
    if Opcode.JMP in runner1._func_call_handlers:
        del runner1._func_call_handlers[Opcode.JMP]

    source1 = """
    int main() {
        return 6 * 7;
    }
    """
    bytecode1, data1 = compile_c(source1)
    _, result1 = runner1.run(bytecode1, data1, max_steps=50)
    print(f"Test 1 (6*7, no JMP): result={result1}, expected=42, {'PASS' if result1 == 42 else 'FAIL'}")

    # Program with conditional (uses BZ/JMP)
    runner2 = AutoregressiveVMRunner()
    if Opcode.JMP in runner2._func_call_handlers:
        del runner2._func_call_handlers[Opcode.JMP]
    if Opcode.BZ in runner2._func_call_handlers:
        del runner2._func_call_handlers[Opcode.BZ]
    if Opcode.BNZ in runner2._func_call_handlers:
        del runner2._func_call_handlers[Opcode.BNZ]

    source2 = """
    int main() {
        if (1) {
            return 42;
        }
        return 99;
    }
    """
    bytecode2, data2 = compile_c(source2)
    _, result2 = runner2.run(bytecode2, data2, max_steps=100)
    print(f"Test 2 (if(1), BZ/JMP): result={result2}, expected=42, {'PASS' if result2 == 42 else 'FAIL'}")

if __name__ == "__main__":
    test()
