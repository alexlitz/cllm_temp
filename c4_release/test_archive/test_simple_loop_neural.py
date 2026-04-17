"""Test a simple loop with the neural VM."""
import time
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

print("Testing simple loop with neural VM...")

# Very simple loop: count to 3
source = """
int main() {
    int i;
    i = 0;
    while (i < 3) {
        i = i + 1;
    }
    return i;
}
"""

print(f"Program: {source.strip()}")

print("\nCompiling...")
bytecode, data = compile_c(source)
print(f"Bytecode: {len(bytecode)} instructions")

print("\nCreating runner...")
runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)
runner.model.eval()

print("\nRunning...")
start = time.time()
output_str, exit_code = runner.run(bytecode, max_steps=50)
elapsed = time.time() - start

print(f"\nResult: exit_code={exit_code} (expected 3)")
print(f"Time: {elapsed:.2f}s")
print(f"Status: {'✓ PASS' if exit_code == 3 else '✗ FAIL'}")
