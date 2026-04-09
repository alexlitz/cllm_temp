"""Simple test - does THINKING_END ever appear?"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = 'int main() { printf("Hi"); return 0; }'

code, data = compile_c(c_code)
runner = AutoregressiveVMRunner(conversational_io=True)

# Track THINKING_END
thinking_end_count = 0

def count_thinking_end(context):
    global thinking_end_count
    token = runner.model.generate_next.__wrapped__(context)
    if token == Token.THINKING_END:
        thinking_end_count += 1
        print(f"[FOUND] THINKING_END at step {thinking_end_count}")
    return token

# Wrap it
original = runner.model.generate_next
runner.model.generate_next.__wrapped__ = original
runner.model.generate_next = count_thinking_end

print("Running...")
output, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\nResult:")
print(f"  Exit code: {exit_code}")
print(f"  THINKING_END count: {thinking_end_count}")
print(f"  Status: {'✅ PASS' if thinking_end_count > 0 else '❌ FAIL'}")
