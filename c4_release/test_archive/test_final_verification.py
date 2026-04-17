#\!/usr/bin/env python3
"""Final verification of complex operations after handler fixes."""
import sys
sys.path.insert(0, '.')
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print("=" * 70)
print("COMPLEX OPERATIONS VERIFICATION TEST")
print("=" * 70)

print("\nInitializing runner...")
runner = AutoregressiveVMRunner()
runner.model.cuda()
print("✓ Runner initialized\n")

# Comprehensive test suite covering all fixed handlers
tests = [
    # Basic operations
    ('int main() { return 42; }', 42, "Basic: Return immediate"),
    ('int main() { return 5 + 3; }', 8, "Basic: Addition"),
    ('int main() { return 10 - 3; }', 7, "Basic: Subtraction"),
    ('int main() { return 6 * 7; }', 42, "Basic: Multiplication"),
    ('int main() { return 20 / 5; }', 4, "Basic: Division"),
    ('int main() { return 17 % 5; }', 2, "Basic: Modulo"),
    
    # Bitwise operations
    ('int main() { return 5 | 3; }', 7, "Bitwise: OR"),
    ('int main() { return 5 & 3; }', 1, "Bitwise: AND"),
    ('int main() { return 5 ^ 3; }', 6, "Bitwise: XOR"),
    ('int main() { return 8 << 2; }', 32, "Bitwise: Left shift"),
    ('int main() { return 32 >> 2; }', 8, "Bitwise: Right shift"),
    
    # Complex: Arithmetic combinations
    ('int main() { return (5 + 3) * 2; }', 16, "Complex: (add) * multiply"),
    ('int main() { return (10 - 3) + 5; }', 12, "Complex: (sub) + add"),
    ('int main() { return 100 / 5 - 10; }', 10, "Complex: (div) - sub"),
    ('int main() { return (3 + 2) * (4 - 1); }', 15, "Complex: (add) * (sub)"),
    
    # Complex: Bitwise combinations
    ('int main() { return (5 | 3) & 7; }', 7, "Complex: (OR) & AND"),
    ('int main() { return (12 ^ 5) | 1; }', 9, "Complex: (XOR) | OR"),
    ('int main() { return (8 << 2) >> 1; }', 16, "Complex: (shl) >> shr"),
    
    # Complex: Mixed operations
    ('int main() { return ((3 + 2) * 4) >> 1; }', 10, "Complex: ((add)*mul)>>shr"),
    ('int main() { return (7 & 15) + 3; }', 10, "Complex: (AND) + add"),
]

print("-" * 70)
print(f"{'Test Description':<40} {'Result':<15} {'Status'}")
print("-" * 70)

passed = 0
failed = 0
errors = 0

for code, expected, desc in tests:
    try:
        bytecode, _ = compile_c(code)
        output, exit_code = runner.run(bytecode, max_steps=300)
        
        if exit_code == expected:
            print(f"{desc:<40} {exit_code:>4} == {expected:<4} ✓")
            passed += 1
        else:
            print(f"{desc:<40} {exit_code:>4} \!= {expected:<4} ✗ FAILED")
            failed += 1
    except Exception as e:
        error_msg = str(e)[:30]
        print(f"{desc:<40} ERROR: {error_msg:<20} ✗")
        errors += 1

print("-" * 70)
print(f"\nRESULTS: {passed} passed, {failed} failed, {errors} errors")
print(f"Success rate: {100*passed/(passed+failed+errors):.1f}%")
print("=" * 70)
