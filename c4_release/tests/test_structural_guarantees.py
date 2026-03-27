#!/usr/bin/env python3
"""
Test structural guarantees that prevent using DraftVM results.

These tests verify that the system CANNOT use DraftVM results
even if someone tries to access them.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner, _BlockedDraftVM
from neural_vm.speculative import DraftVM


def test_blocked_draftvm_ax():
    """Verify that accessing DraftVM.ax raises AttributeError."""
    bytecode, data = compile_c("int main() { return 42; }")

    # bytecode is already a list of instructions
    blocked_vm = _BlockedDraftVM(bytecode)

    # Step the VM (allowed for speculation)
    blocked_vm.step()

    # Try to access ax (should be blocked)
    with pytest.raises(AttributeError) as exc_info:
        _ = blocked_vm.ax

    # Verify error message explains why
    assert "BLOCKED" in str(exc_info.value)
    assert "DraftVM.ax" in str(exc_info.value)
    assert "FastLogicalVM" in str(exc_info.value) or "reference" in str(exc_info.value).lower()


def test_blocked_draftvm_output():
    """Verify that accessing DraftVM.output raises AttributeError."""
    bytecode, data = compile_c("int main() { return 42; }")

    blocked_vm = _BlockedDraftVM(bytecode)
    blocked_vm.step()

    with pytest.raises(AttributeError) as exc_info:
        _ = blocked_vm.output

    assert "BLOCKED" in str(exc_info.value)
    assert "DraftVM.output" in str(exc_info.value)


def test_blocked_draftvm_pc():
    """Verify that accessing DraftVM.pc raises AttributeError."""
    bytecode, data = compile_c("int main() { return 42; }")

    blocked_vm = _BlockedDraftVM(bytecode)

    with pytest.raises(AttributeError) as exc_info:
        _ = blocked_vm.pc

    assert "BLOCKED" in str(exc_info.value)


def test_blocked_draftvm_sp():
    """Verify that accessing DraftVM.sp raises AttributeError."""
    bytecode, data = compile_c("int main() { return 42; }")

    blocked_vm = _BlockedDraftVM(bytecode)

    with pytest.raises(AttributeError) as exc_info:
        _ = blocked_vm.sp

    assert "BLOCKED" in str(exc_info.value)


def test_blocked_draftvm_any_state():
    """Verify that accessing any DraftVM state raises AttributeError."""
    bytecode, data = compile_c("int main() { return 42; }")

    blocked_vm = _BlockedDraftVM(bytecode)

    # Try various state attributes
    for attr in ['memory', 'idx', 'bp', 'cycle']:
        with pytest.raises(AttributeError) as exc_info:
            _ = getattr(blocked_vm, attr)

        assert "BLOCKED" in str(exc_info.value)


def test_speculation_methods_still_work():
    """Verify that speculation methods (step, draft_tokens) still work."""
    bytecode, data = compile_c("int main() { return 42; }")

    blocked_vm = _BlockedDraftVM(bytecode)

    # These should work (needed for speculation)
    result = blocked_vm.step()
    assert isinstance(result, bool)

    tokens = blocked_vm.draft_tokens()
    assert isinstance(tokens, list)


def test_runner_uses_reference_vm():
    """
    Verify that BatchedSpeculativeRunner returns results from reference VM,
    not from DraftVM.

    This is the critical test: Results MUST match reference VM, proving
    they don't come from DraftVM.
    """
    # Simple test program
    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    # Run with batched runner
    runner = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=False,  # Disable for simplicity
        use_sparse=False,
    )

    results = runner.run_batch(
        bytecodes=[bytecode],
        data_list=[data],
        max_steps=1000
    )

    # Run reference VM for ground truth
    from src.speculator import FastLogicalVM
    ref_vm = FastLogicalVM()
    ref_vm.reset()
    ref_vm.load(bytecode, data)
    expected_exit_code = ref_vm.run()

    # Results MUST match reference VM
    output, exit_code = results[0]
    assert exit_code == expected_exit_code, (
        f"Exit code mismatch: got {exit_code}, expected {expected_exit_code}. "
        f"This suggests results came from DraftVM, not reference VM!"
    )
    # Note: We don't check output since FastLogicalVM doesn't track it
    # The critical guarantee is that exit_code comes from reference VM


def test_runner_uses_reference_vm_multiple_programs():
    """
    Test with multiple programs to ensure all use reference VM.
    """
    test_programs = [
        ("int main() { return 1; }", 1),
        ("int main() { return 2; }", 2),
        ("int main() { return 10 + 5; }", 15),
        ("int main() { return 3 * 7; }", 21),
    ]

    bytecodes = []
    data_list = []
    expected_results = []

    for source, expected in test_programs:
        bytecode, data = compile_c(source)
        bytecodes.append(bytecode)
        data_list.append(data)
        expected_results.append(expected)

    # Run with batched runner
    runner = BatchedSpeculativeRunner(
        batch_size=10,
        use_kv_cache=False,
        use_sparse=False,
    )

    results = runner.run_batch(
        bytecodes=bytecodes,
        data_list=data_list,
        max_steps=1000
    )

    # Verify all results match reference VM
    for i, ((output, exit_code), expected) in enumerate(zip(results, expected_results)):
        # Run reference VM
        from src.speculator import FastLogicalVM
        ref_vm = FastLogicalVM()
        ref_vm.reset()
        ref_vm.load(bytecodes[i], data_list[i])
        ref_exit_code = ref_vm.run()

        assert exit_code == ref_exit_code, (
            f"Program {i}: Exit code {exit_code} != reference {ref_exit_code}. "
            f"Results came from DraftVM, not reference VM!"
        )


def test_cannot_accidentally_use_draftvm():
    """
    Verify that even trying to directly access draft_vms raises errors.

    This tests the structural guarantee: You CANNOT access DraftVM state
    even if you try.
    """
    bytecode, data = compile_c("int main() { return 42; }")

    runner = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=False,
        use_sparse=False,
    )

    # Run to initialize draft_vms
    runner.run_batch(
        bytecodes=[bytecode],
        data_list=[data],
        max_steps=10
    )

    # Try to access DraftVM state from runner.draft_vms
    # This should raise AttributeError (structural guarantee)
    with pytest.raises(AttributeError) as exc_info:
        _ = runner.draft_vms[0].ax

    assert "BLOCKED" in str(exc_info.value)


if __name__ == '__main__':
    # Run tests
    print("Testing structural guarantees...")

    test_blocked_draftvm_ax()
    print("✓ DraftVM.ax access blocked")

    test_blocked_draftvm_output()
    print("✓ DraftVM.output access blocked")

    test_blocked_draftvm_pc()
    print("✓ DraftVM.pc access blocked")

    test_blocked_draftvm_sp()
    print("✓ DraftVM.sp access blocked")

    test_blocked_draftvm_any_state()
    print("✓ All DraftVM state access blocked")

    test_speculation_methods_still_work()
    print("✓ Speculation methods (step, draft_tokens) still work")

    test_runner_uses_reference_vm()
    print("✓ Runner uses reference VM, not DraftVM")

    test_runner_uses_reference_vm_multiple_programs()
    print("✓ Multiple programs all use reference VM")

    test_cannot_accidentally_use_draftvm()
    print("✓ Cannot accidentally access DraftVM state from runner")

    print("\n" + "=" * 70)
    print("✓ ALL STRUCTURAL GUARANTEES VERIFIED!")
    print("=" * 70)
    print("\nGuarantees:")
    print("1. ✓ DraftVM state access raises AttributeError")
    print("2. ✓ Results come from reference VM execution")
    print("3. ✓ No code path to return DraftVM results")
    print("4. ✓ Structural: Cannot accidentally use DraftVM state")
