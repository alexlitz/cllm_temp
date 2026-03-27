"""Test that longer programs work with the neural VM after PC_OFFSET=0 fix."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from src.compiler import compile_c

def test_neural_runner(name, bytecode, expected_exit, max_steps=100):
    """Test a program using the actual neural AutoregressiveVMRunner."""
    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    runner.model.eval()

    output_str, exit_code = runner.run(bytecode, max_steps=max_steps)

    status = "✓" if exit_code == expected_exit else "✗"
    print(f"{status} {name}: exit={exit_code}, expected={expected_exit}, steps≤{max_steps}")

    return exit_code == expected_exit

print("Testing longer neural programs after PC_OFFSET=0 fix:")
print("=" * 70)

all_pass = True

# Test 1: Simple loop (count to 10)
print("\n1. Simple counting loop (10 iterations):")
source = """
int main() {
    int i;
    i = 0;
    while (i < 10) {
        i = i + 1;
    }
    return i;
}
"""
bytecode, data = compile_c(source)
all_pass &= test_neural_runner("Count to 10", bytecode, 10, max_steps=100)

# Test 2: Iterative factorial
print("\n2. Factorial calculation:")
source = """
int main() {
    int n, result, i;
    n = 5;
    result = 1;
    i = 1;
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    return result;
}
"""
bytecode, data = compile_c(source)
all_pass &= test_neural_runner("Factorial(5)", bytecode, 120, max_steps=100)

# Test 3: Fibonacci (iterative)
print("\n3. Fibonacci calculation:")
source = """
int main() {
    int n, a, b, i, tmp;
    n = 10;
    if (n < 2) return n;
    a = 0;
    b = 1;
    i = 2;
    while (i <= n) {
        tmp = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}
"""
bytecode, data = compile_c(source)
all_pass &= test_neural_runner("Fibonacci(10)", bytecode, 55, max_steps=150)

# Test 4: Summation loop
print("\n4. Sum 1 to 20:")
source = """
int main() {
    int i, sum;
    i = 1;
    sum = 0;
    while (i <= 20) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
"""
bytecode, data = compile_c(source)
all_pass &= test_neural_runner("Sum(1..20)", bytecode, 210, max_steps=150)

# Test 5: Nested loops
print("\n5. Nested loops (3x3 grid):")
source = """
int main() {
    int i, j, count;
    count = 0;
    i = 0;
    while (i < 3) {
        j = 0;
        while (j < 3) {
            count = count + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return count;
}
"""
bytecode, data = compile_c(source)
all_pass &= test_neural_runner("Nested 3x3", bytecode, 9, max_steps=200)

print("\n" + "=" * 70)
if all_pass:
    print("🎉 ALL LONGER NEURAL PROGRAMS PASSED!")
    print("PC_OFFSET=0 conversion is fully functional for complex programs.")
else:
    print("❌ Some longer programs failed")
    print("Need to investigate neural VM behavior for longer programs.")
