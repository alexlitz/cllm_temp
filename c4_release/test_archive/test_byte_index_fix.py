"""Test if hop-count fix resolves BYTE_INDEX bug in L14 MEM generation."""

import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

# Simple JSR test program
code = '''
int helper() { return 42; }
int main() {
    helper();
    return 0;
}
'''

print('Compiling code...')
bytecode, data = compile_c(code)
print(f'✓ Compiled: {len(bytecode)} bytes of bytecode')

print('\nCreating fresh runner (forces weight reinitialization)...')
# Creating a new runner forces set_vm_weights() to run with updated code
runner = AutoregressiveVMRunner(debug_memory=True)

print('\nRunning VM to test JSR MEM address generation...')
print('=' * 70)
print('Expected: MEM addresses with bytes [X, Y, Z, 0] (byte 3 should be 0)')
print('Bug: MEM addresses with bytes [X, Y, Z, X] (byte 3 copies byte 0)')
print('=' * 70)

try:
    output = runner.run(bytecode, data, max_steps=10)
    print(f'\n✓ Execution completed, exit code: {output}')
    print('\n[ANALYSIS]')
    print('If you see addresses like 0xf80001f8 (byte 3 = f8 = byte 0), bug still present.')
    print('If you see addresses like 0x000001f8 (byte 3 = 00), bug is FIXED!')
except Exception as e:
    print(f'\n✗ Execution failed: {e}')
    import traceback
    traceback.print_exc()
