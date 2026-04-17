"""Test JSR with MEM token debugging."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

code = '''
int helper() { return 42; }
int main() { return helper(); }
'''

print('Testing JSR with MEM debug...')
print('=' * 70)

print('\n1. Compiling...')
bytecode, data = compile_c(code)
print(f'   ✓ Compiled')

print('\n2. Running with MEM tracking...')

# Patch runner to capture MEM tokens
mem_tokens_found = []

runner = AutoregressiveVMRunner()

# Patch the _extract_mem_section method to capture MEM tokens
original_extract = runner._extract_mem_section

def debug_extract(context):
    # Find MEM tokens in context
    for i in range(len(context)):
        if context[i] == Token.MEM and i + 8 < len(context):
            addr_bytes = [context[i+1], context[i+2], context[i+3], context[i+4]]
            val_bytes = [context[i+5], context[i+6], context[i+7], context[i+8]]
            mem_tokens_found.append({
                'position': i,
                'addr': addr_bytes,
                'val': val_bytes
            })
    return original_extract(context)

runner._extract_mem_section = debug_extract

try:
    result = runner.run(bytecode, data, max_steps=20)
    print(f'   Result: {result}')
    print(f'   Expected: 42')

    if mem_tokens_found:
        print(f'\n3. Found {len(mem_tokens_found)} MEM token(s):')
        for idx, mem in enumerate(mem_tokens_found[:3]):  # Show first 3
            addr_hex = ''.join(f'{b:02x}' for b in mem['addr'])
            val_hex = ''.join(f'{b:02x}' for b in mem['val'])
            print(f'\n   MEM #{idx+1} (pos {mem["position"]}):')
            print(f'     Addr: {mem["addr"]} → 0x{addr_hex}')
            print(f'     Val:  {mem["val"]} → 0x{val_hex}')

            # Check for addr bug
            if mem['addr'][3] == mem['addr'][0] and mem['addr'][0] != 0:
                print(f'     ⚠️  Addr byte 3 copies byte 0!')
            # Check for val issues
            if mem['val'] == [0, 0, 0, 0]:
                print(f'     ⚠️  Val is all zeros!')
    else:
        print('\n⚠️  No MEM tokens found!')

except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()

print('=' * 70)
