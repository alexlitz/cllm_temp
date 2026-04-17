#!/usr/bin/env python3
"""
Simple test to verify basic VM functionality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.speculative import DraftVM
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

def test_simple_execution():
    """Test simple program execution with DraftVM."""
    print("=" * 60)
    print("TEST: Simple VM Execution")
    print("=" * 60)

    # Simple test program
    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    print(f"\nSource: {source}")
    print(f"Bytecode length: {len(bytecode)} instructions")
    print()

    # Test DraftVM
    print("1. Testing DraftVM (fast logical VM)...")
    draft_vm = DraftVM(bytecode)
    while draft_vm.step():
        pass
    result_draft = draft_vm.ax
    print(f"   Result: {result_draft}")
    print(f"   Expected: 42")
    print(f"   ✓ PASS" if result_draft == 42 else f"   ✗ FAIL")
    print()

    return 0 if result_draft == 42 else 1


if __name__ == '__main__':
    sys.exit(test_simple_execution())
