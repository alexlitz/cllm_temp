"""
Quick test if ADD works without handler, despite Head 6 conflict.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import torch

print("Testing ADD operation (10 + 32 = 42)...\n")

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

# Test without handler
runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("Removed ADD handler - using neural implementation only\n")

try:
    output, exit_code = runner.run(bytecode, max_steps=15)
    print(f"Result: {output}")
    print(f"Exit code: {exit_code}\n")

    if output == 42 and exit_code == 0:
        print("✓✓✓ SUCCESS! ADD returns 42!")
    else:
        print(f"❌ FAIL: Expected 42, got {output}")
except Exception as e:
    print(f"❌ ERROR: {e}")

torch.cuda.empty_cache()
