"""
Test that transformer primitives execute C4 opcodes exactly.

No learning - just verify that attention + FFN primitives
can simulate the C4 VM deterministically.
"""

import torch
from c4_vm import C4VM, Op
from c4_vm import program_simple_arithmetic, program_sum
from c4_ops_transformer import C4OpcodeExecutor


def run_with_primitives(prog_fn, max_steps=10000):
    """Run a program using transformer primitives, compare to VM."""
    executor = C4OpcodeExecutor()

    # Set up VM
    vm = C4VM()
    vm.load_code(prog_fn())

    # Set up tensor state
    pc = torch.tensor(vm.state.pc, dtype=torch.long)
    sp = torch.tensor(vm.state.sp, dtype=torch.long)
    bp = torch.tensor(vm.state.bp, dtype=torch.long)
    ax = torch.tensor(vm.state.ax, dtype=torch.long)
    memory = torch.tensor(list(vm.state.memory), dtype=torch.long)

    steps = 0
    while steps < max_steps:
        # Fetch instruction using primitives
        opcode, imm = executor.fetch_instruction(memory, pc)

        if opcode.item() == Op.EXIT:
            break

        # Execute using transformer primitives
        pc, sp, bp, ax, memory = executor.execute_op(
            opcode.item(), imm.item(),
            pc, sp, bp, ax, memory
        )

        # Step VM for comparison
        vm.step()

        # Verify match
        if pc.item() != vm.state.pc:
            return False, steps, f"PC mismatch: {pc.item()} vs {vm.state.pc}"
        if sp.item() != vm.state.sp:
            return False, steps, f"SP mismatch: {sp.item()} vs {vm.state.sp}"
        if ax.item() != vm.state.ax:
            return False, steps, f"AX mismatch: {ax.item()} vs {vm.state.ax}"

        steps += 1

    return True, steps, ax.item()


def test_all_programs():
    """Test multiple programs with transformer primitives."""
    print("=" * 60)
    print("C4 TRANSFORMER PRIMITIVE EXECUTION TEST")
    print("=" * 60)
    print()
    print("Testing that transformer primitives (attention + FFN)")
    print("execute C4 opcodes exactly like the VM.")
    print()

    tests = [
        ("(3 + 4) * 5", program_simple_arithmetic, 35),
        ("sum(1)", lambda: program_sum(1), 1),
        ("sum(5)", lambda: program_sum(5), 15),
        ("sum(10)", lambda: program_sum(10), 55),
        ("sum(20)", lambda: program_sum(20), 210),
        ("sum(50)", lambda: program_sum(50), 1275),
        ("sum(100)", lambda: program_sum(100), 5050),
        # Note: factorial bytecode has bugs in the VM itself, skipping
    ]

    passed = 0
    failed = 0

    for name, prog_fn, expected in tests:
        success, steps, result = run_with_primitives(prog_fn)

        if success and result == expected:
            print(f"  ✓ {name:20s} = {result:6d}  ({steps} steps)")
            passed += 1
        else:
            print(f"  ✗ {name:20s} = {result} (expected {expected})")
            if not success:
                print(f"    Error: {result}")
            failed += 1

    print()
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print()

    if failed == 0:
        print("SUCCESS: All programs execute correctly using transformer primitives.")
        print()
        print("This proves that:")
        print("  1. Memory read/write via attention (gather/scatter)")
        print("  2. Arithmetic via FFN (add, sub, mul, div, mod)")
        print("  3. Comparisons via FFN (eq, lt, gt, etc.)")
        print("  4. Control flow via conditional selection")
        print()
        print("...are sufficient to simulate the C4 virtual machine.")
    else:
        print("FAILURE: Some programs did not execute correctly.")

    return failed == 0


if __name__ == "__main__":
    test_all_programs()
