#!/usr/bin/env python3
"""Test JSR after ALiBi fix - should work now!"""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

# Simple function call
code = '''
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(10, 32);
}
'''

bytecode, data = compile_c(code)
print(f"Bytecode length: {len(bytecode)} instructions")
print(f"Bytecode[0] = 0x{bytecode[0]:08x} (opcode {bytecode[0] & 0xFF})")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # Disable all handlers for 100% neural
result = runner.run(bytecode, data, [], "")

# Result is (output, exit_code)
output, exit_code = result if isinstance(result, tuple) else ("", result)
print(f"\nResult: exit_code={exit_code}, output='{output}'")
print(f"Expected: 42 (10 + 32)")
print(f"Correct: {exit_code == 42}")
