"""Test opcodes with handlers disabled to see if neural weights work"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import torch
import sys

def test_with_handlers_disabled(code, expected, desc, opcodes_to_disable):
    """Test with specific handlers disabled."""
    print(f"\n{desc}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)

        runner = AutoregressiveVMRunner()

        # Disable specified handlers
        for op in opcodes_to_disable:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]
                print(f"  Disabled handler for opcode {op}", file=sys.stderr)

        runner.model.cuda()
        _, exit_code = runner.run(bytecode, max_steps=100)

        del runner
        torch.cuda.empty_cache()

        if exit_code == expected:
            print(f"  ✓ PASS: exit_code={exit_code}", file=sys.stderr)
            return True
        else:
            print(f"  ✗ FAIL: exit_code={exit_code}, expected {expected}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:80]}", file=sys.stderr)
        torch.cuda.empty_cache()
        return False

print("=" * 70, file=sys.stderr)
print("TESTING OPCODES WITH HANDLERS DISABLED", file=sys.stderr)
print("=" * 70, file=sys.stderr)

passed = failed = 0

# Test ADD with handler disabled
print("\n--- ARITHMETIC OPS (disable ADD handler) ---", file=sys.stderr)
if test_with_handlers_disabled("int main() { return 10 + 32; }", 42, "ADD (no handler)", [Opcode.ADD]):
    passed += 1
else:
    failed += 1

# Test SUB with handler disabled
if test_with_handlers_disabled("int main() { return 50 - 8; }", 42, "SUB (no handler)", [Opcode.SUB]):
    passed += 1
else:
    failed += 1

# Test bitwise ops with handlers disabled
print("\n--- BITWISE OPS (disable handlers) ---", file=sys.stderr)
if test_with_handlers_disabled("int main() { return 32 | 10; }", 42, "OR (no handler)", [Opcode.OR]):
    passed += 1
else:
    failed += 1

if test_with_handlers_disabled("int main() { return 40 ^ 2; }", 42, "XOR (no handler)", [Opcode.XOR]):
    passed += 1
else:
    failed += 1

if test_with_handlers_disabled("int main() { return 63 & 42; }", 42, "AND (no handler)", [Opcode.AND]):
    passed += 1
else:
    failed += 1

# Test shift ops with handlers disabled
print("\n--- SHIFT OPS (disable handlers) ---", file=sys.stderr)
if test_with_handlers_disabled("int main() { return 21 << 1; }", 42, "SHL (no handler)", [Opcode.SHL]):
    passed += 1
else:
    failed += 1

if test_with_handlers_disabled("int main() { return 84 >> 1; }", 42, "SHR (no handler)", [Opcode.SHR]):
    passed += 1
else:
    failed += 1

# Note: EQ and LT don't have handlers, so purity violations must come from elsewhere
print("\n--- COMPARISON OPS (no handlers to disable) ---", file=sys.stderr)
print("Note: EQ and LT don't have handlers in _func_call_handlers", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)
print(f"RESULTS: {passed}/{passed+failed} passed", file=sys.stderr)
if failed > 0:
    print(f"FAILED: {failed} tests", file=sys.stderr)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
print("=" * 70, file=sys.stderr)
