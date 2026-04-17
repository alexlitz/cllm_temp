"""Test JSR after val heads revert."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = '''
int helper() { return 42; }
int main() { return helper(); }
'''

print('Testing JSR after val heads revert...')
print('=' * 70)

print('\n1. Compiling...')
bytecode, data = compile_c(code)
print(f'   ✓ Compiled: {len(bytecode)} bytes')

print('\n2. Running in hybrid mode (max 20 steps)...')
try:
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, max_steps=20)
    print(f'   Result: {result}')
    print(f'   Expected: 42')

    if result == 42:
        print('   ✓ TEST PASSED!')
    else:
        print(f'   ✗ TEST FAILED: Expected 42, got {result}')

except Exception as e:
    print(f'   ✗ Error during execution: {e}')
    import traceback
    traceback.print_exc()

print('=' * 70)
