"""Tests for unified memory - executing code written to memory.

Tests DraftVM's ability to fetch and execute instructions from memory,
enabling self-modifying code and JIT compilation patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.speculative import DraftVM
from neural_vm.constants import INSTR_WIDTH


# Opcode constants
OP_LEA = 0
OP_IMM = 1
OP_JMP = 2
OP_JSR = 3
OP_BZ = 4
OP_BNZ = 5
OP_ENT = 6
OP_ADJ = 7
OP_LEV = 8
OP_LI = 9
OP_SI = 11
OP_PSH = 13
OP_ADD = 25
OP_SUB = 26
OP_MUL = 27
OP_EXIT = 38


def pack_instr(op, imm=0):
    """Pack opcode + 24-bit immediate into 32-bit instruction."""
    return op | ((imm & 0xFFFFFF) << 8)


def test_draftvm_jump_to_data_region():
    """Test jumping to code written in the data region (0x10000+)."""
    # Static code: JMP to data region
    data_addr = 0x10000
    code = [
        pack_instr(OP_JMP, data_addr),  # JMP to data region
    ]
    vm = DraftVM(code)

    # Write code to data region: IMM 42, EXIT
    vm.write_code_to_memory(data_addr, OP_IMM, 42)
    vm.write_code_to_memory(data_addr + 4, OP_EXIT, 0)

    # Execute JMP
    assert vm.step(), "JMP should execute"
    assert vm.pc == data_addr, f"PC should be at data region, got {vm.pc}"

    # Execute IMM 42 from memory
    assert vm.step(), "IMM from memory should execute"
    assert vm.ax == 42, f"AX should be 42, got {vm.ax}"

    # Execute EXIT from memory
    assert vm.step(), "EXIT from memory should execute"
    assert vm.halted, "VM should be halted"

    print("PASS: test_draftvm_jump_to_data_region")


def test_draftvm_jsr_to_data_and_return():
    """Test JSR to code in data region, then LEV to return."""
    data_addr = 0x10000
    code = [
        pack_instr(OP_IMM, 10),          # 0: AX = 10
        pack_instr(OP_JSR, data_addr),   # 1: Call function in data region
        pack_instr(OP_EXIT, 0),          # 2: EXIT after return
    ]
    vm = DraftVM(code)

    # Write function to data region: add 32 to AX, return
    # ENT 0, PSH, IMM 32, ADD, LEV
    vm.write_code_to_memory(data_addr, OP_ENT, 0)
    vm.write_code_to_memory(data_addr + 4, OP_PSH, 0)
    vm.write_code_to_memory(data_addr + 8, OP_IMM, 32)
    vm.write_code_to_memory(data_addr + 12, OP_ADD, 0)
    vm.write_code_to_memory(data_addr + 16, OP_LEV, 0)

    # Execute: IMM 10
    vm.step()
    assert vm.ax == 10

    # Execute: JSR to data region
    vm.step()
    assert vm.pc == data_addr

    # Execute function in data region
    vm.step()  # ENT 0
    vm.step()  # PSH (push AX=10)
    vm.step()  # IMM 32
    assert vm.ax == 32
    vm.step()  # ADD (10 + 32 = 42)
    assert vm.ax == 42

    # LEV should return to static code
    vm.step()  # LEV

    # Should be back in static code at EXIT
    vm.step()  # EXIT
    assert vm.halted
    assert vm.ax == 42, f"AX should be 42 after function call, got {vm.ax}"

    print("PASS: test_draftvm_jsr_to_data_and_return")


def test_draftvm_conditional_jump_to_data():
    """Test BZ/BNZ jumping to code in data region."""
    data_addr = 0x10000
    code = [
        pack_instr(OP_IMM, 0),           # 0: AX = 0
        pack_instr(OP_BZ, data_addr),    # 1: BZ to data (should jump since AX=0)
        pack_instr(OP_IMM, 99),          # 2: Should not reach
        pack_instr(OP_EXIT, 0),          # 3: Should not reach
    ]
    vm = DraftVM(code)

    # Write code to data region: IMM 42, EXIT
    vm.write_code_to_memory(data_addr, OP_IMM, 42)
    vm.write_code_to_memory(data_addr + 4, OP_EXIT, 0)

    # Execute IMM 0
    vm.step()
    assert vm.ax == 0

    # Execute BZ - should jump to data region
    vm.step()
    assert vm.pc == data_addr, f"BZ should jump to data region, PC={vm.pc}"

    # Execute IMM 42 from data region
    vm.step()
    assert vm.ax == 42, f"AX should be 42, got {vm.ax}"

    # Execute EXIT
    vm.step()
    assert vm.halted

    print("PASS: test_draftvm_conditional_jump_to_data")


def test_draftvm_bnz_to_data():
    """Test BNZ jumping to code in data region."""
    data_addr = 0x10000
    code = [
        pack_instr(OP_IMM, 1),           # 0: AX = 1 (non-zero)
        pack_instr(OP_BNZ, data_addr),   # 1: BNZ to data (should jump since AX!=0)
        pack_instr(OP_EXIT, 0),          # 2: Should not reach
    ]
    vm = DraftVM(code)

    # Write code to data region
    vm.write_code_to_memory(data_addr, OP_IMM, 100)
    vm.write_code_to_memory(data_addr + 4, OP_EXIT, 0)

    vm.step()  # IMM 1
    vm.step()  # BNZ - should jump
    assert vm.pc == data_addr

    vm.step()  # IMM 100
    assert vm.ax == 100

    vm.step()  # EXIT
    assert vm.halted

    print("PASS: test_draftvm_bnz_to_data")


def test_draftvm_loop_in_data_region():
    """Test executing a loop written in the data region."""
    data_addr = 0x10000
    code = [
        pack_instr(OP_IMM, 5),           # 0: AX = 5 (counter)
        pack_instr(OP_JMP, data_addr),   # 1: Jump to loop in data
    ]
    vm = DraftVM(code)

    # Write loop to data region:
    # loop: PSH, IMM 1, SUB, BNZ loop, EXIT
    loop_addr = data_addr
    vm.write_code_to_memory(loop_addr, OP_PSH, 0)        # Push counter
    vm.write_code_to_memory(loop_addr + 4, OP_IMM, 1)    # 1
    vm.write_code_to_memory(loop_addr + 8, OP_SUB, 0)    # counter - 1
    vm.write_code_to_memory(loop_addr + 12, OP_BNZ, loop_addr)  # loop if non-zero
    vm.write_code_to_memory(loop_addr + 16, OP_EXIT, 0)  # exit when done

    # Run until halted
    steps = 0
    while vm.step() and steps < 100:
        steps += 1

    assert vm.halted, "VM should halt after loop"
    assert vm.ax == 0, f"AX should be 0 after countdown, got {vm.ax}"
    # 5 iterations: each iteration is PSH, IMM, SUB, BNZ = 4 steps, plus final EXIT
    # Plus initial IMM and JMP = 2 steps
    # Total: 2 + 5*4 + 1 (final PSH,IMM,SUB,BZ-not-taken) + EXIT = 2 + 20 + 1 = 23?
    # Actually: iterations decrement 5->4->3->2->1->0
    # 5 iterations with BNZ taken, then BNZ not taken + EXIT
    print(f"PASS: test_draftvm_loop_in_data_region (completed in {steps} steps)")


def test_draftvm_arithmetic_in_data():
    """Test arithmetic operations in code written to data region."""
    data_addr = 0x10000
    code = [
        pack_instr(OP_JMP, data_addr),
    ]
    vm = DraftVM(code)

    # Write: IMM 10, PSH, IMM 7, MUL (10 * 7 = 70), EXIT
    vm.write_code_to_memory(data_addr, OP_IMM, 10)
    vm.write_code_to_memory(data_addr + 4, OP_PSH, 0)
    vm.write_code_to_memory(data_addr + 8, OP_IMM, 7)
    vm.write_code_to_memory(data_addr + 12, OP_MUL, 0)
    vm.write_code_to_memory(data_addr + 16, OP_EXIT, 0)

    while vm.step():
        pass

    assert vm.ax == 70, f"10 * 7 should be 70, got {vm.ax}"
    print("PASS: test_draftvm_arithmetic_in_data")


def test_draftvm_multiple_functions_in_data():
    """Test calling multiple functions written to data region."""
    fn1_addr = 0x10000
    fn2_addr = 0x10020
    code = [
        pack_instr(OP_IMM, 5),           # AX = 5
        pack_instr(OP_JSR, fn1_addr),    # Call fn1 (doubles AX)
        pack_instr(OP_JSR, fn2_addr),    # Call fn2 (adds 10)
        pack_instr(OP_EXIT, 0),
    ]
    vm = DraftVM(code)

    # fn1: doubles AX - ENT, PSH, IMM 2, MUL, LEV
    vm.write_code_to_memory(fn1_addr, OP_ENT, 0)
    vm.write_code_to_memory(fn1_addr + 4, OP_PSH, 0)
    vm.write_code_to_memory(fn1_addr + 8, OP_IMM, 2)
    vm.write_code_to_memory(fn1_addr + 12, OP_MUL, 0)
    vm.write_code_to_memory(fn1_addr + 16, OP_LEV, 0)

    # fn2: adds 10 - ENT, PSH, IMM 10, ADD, LEV
    vm.write_code_to_memory(fn2_addr, OP_ENT, 0)
    vm.write_code_to_memory(fn2_addr + 4, OP_PSH, 0)
    vm.write_code_to_memory(fn2_addr + 8, OP_IMM, 10)
    vm.write_code_to_memory(fn2_addr + 12, OP_ADD, 0)
    vm.write_code_to_memory(fn2_addr + 16, OP_LEV, 0)

    while vm.step():
        pass

    # 5 * 2 + 10 = 20
    assert vm.ax == 20, f"5 * 2 + 10 should be 20, got {vm.ax}"
    print("PASS: test_draftvm_multiple_functions_in_data")


def test_draftvm_nested_calls_in_data():
    """Test nested function calls where both functions are in data region."""
    outer_fn = 0x10000
    inner_fn = 0x10030
    code = [
        pack_instr(OP_IMM, 3),
        pack_instr(OP_JSR, outer_fn),
        pack_instr(OP_EXIT, 0),
    ]
    vm = DraftVM(code)

    # outer_fn: calls inner_fn, then adds 100
    # ENT, PSH, JSR inner_fn, PSH, IMM 100, ADD, LEV
    vm.write_code_to_memory(outer_fn, OP_ENT, 0)
    vm.write_code_to_memory(outer_fn + 4, OP_PSH, 0)
    vm.write_code_to_memory(outer_fn + 8, OP_JSR, inner_fn)
    vm.write_code_to_memory(outer_fn + 12, OP_PSH, 0)
    vm.write_code_to_memory(outer_fn + 16, OP_IMM, 100)
    vm.write_code_to_memory(outer_fn + 20, OP_ADD, 0)
    vm.write_code_to_memory(outer_fn + 24, OP_LEV, 0)

    # inner_fn: multiplies by 10
    vm.write_code_to_memory(inner_fn, OP_ENT, 0)
    vm.write_code_to_memory(inner_fn + 4, OP_PSH, 0)
    vm.write_code_to_memory(inner_fn + 8, OP_IMM, 10)
    vm.write_code_to_memory(inner_fn + 12, OP_MUL, 0)
    vm.write_code_to_memory(inner_fn + 16, OP_LEV, 0)

    while vm.step():
        pass

    # 3 * 10 + 100 = 130
    assert vm.ax == 130, f"3 * 10 + 100 should be 130, got {vm.ax}"
    print("PASS: test_draftvm_nested_calls_in_data")


if __name__ == "__main__":
    test_draftvm_jump_to_data_region()
    test_draftvm_jsr_to_data_and_return()
    test_draftvm_conditional_jump_to_data()
    test_draftvm_bnz_to_data()
    test_draftvm_loop_in_data_region()
    test_draftvm_arithmetic_in_data()
    test_draftvm_multiple_functions_in_data()
    test_draftvm_nested_calls_in_data()
    print("\nAll unified memory tests passed!")
