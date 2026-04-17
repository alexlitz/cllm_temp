#!/usr/bin/env python3
"""End-to-end tests for PSH and ADJ opcodes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner


def make_runner():
    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    return runner


def test_imm_exit():
    """Baseline: IMM + EXIT."""
    print("=== Test 1: IMM 42 + EXIT ===")
    runner = make_runner()
    program = [
        Opcode.IMM | (42 << 8),
        Opcode.EXIT,
    ]
    output, exit_code = runner.run(program, max_steps=20)
    print(f"  Exit code: {exit_code} (expected 42)")
    assert exit_code == 42, f"FAIL: expected 42 got {exit_code}"
    print("  PASS")


def test_psh_add():
    """PSH + ADD: push 5, load 3, add → 8."""
    print("\n=== Test 2: IMM 5; PSH; IMM 3; ADD; EXIT ===")
    runner = make_runner()
    program = [
        Opcode.IMM | (5 << 8),
        Opcode.PSH,
        Opcode.IMM | (3 << 8),
        Opcode.ADD,
        Opcode.EXIT,
    ]
    output, exit_code = runner.run(program, max_steps=20)
    print(f"  Exit code: {exit_code} (expected 8)")
    assert exit_code == 8, f"FAIL: expected 8 got {exit_code}"
    print("  PASS")


def test_adj():
    """ADJ: push twice then adjust stack."""
    print("\n=== Test 3: PSH + PSH + ADJ 16 ===")
    runner = make_runner()
    program = [
        Opcode.IMM | (10 << 8),
        Opcode.PSH,
        Opcode.IMM | (20 << 8),
        Opcode.PSH,
        Opcode.ADJ | (16 << 8),   # pop 2 values
        Opcode.IMM | (99 << 8),
        Opcode.EXIT,
    ]
    output, exit_code = runner.run(program, max_steps=20)
    print(f"  Exit code: {exit_code} (expected 99)")
    assert exit_code == 99, f"FAIL: expected 99 got {exit_code}"
    print("  PASS")


def test_psh_sub():
    """PSH + SUB: push 10, load 3, sub → 7."""
    print("\n=== Test 4: IMM 10; PSH; IMM 3; SUB; EXIT ===")
    runner = make_runner()
    program = [
        Opcode.IMM | (10 << 8),
        Opcode.PSH,
        Opcode.IMM | (3 << 8),
        Opcode.SUB,
        Opcode.EXIT,
    ]
    output, exit_code = runner.run(program, max_steps=20)
    print(f"  Exit code: {exit_code} (expected 7)")
    assert exit_code == 7, f"FAIL: expected 7 got {exit_code}"
    print("  PASS")


if __name__ == "__main__":
    test_imm_exit()
    test_psh_add()
    test_adj()
    test_psh_sub()
    print("\nAll tests passed!")
