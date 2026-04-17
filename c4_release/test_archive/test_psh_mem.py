"""Test PSH MEM generation (simpler than JSR)."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

code = '''
int main() {
    return 42;
}
'''

print('Testing PSH MEM generation...')
print('=' * 70)

print('\n1. Compiling...')
bytecode, data = compile_c(code)
print(f'   ✓ Compiled')

print('\n2. Running...')

# Patch runner to capture first few MEM tokens
mem_tokens_found = []

runner = AutoregressiveVMRunner()
original_extract = runner._extract_mem_section

def debug_extract(context):
    if len(mem_tokens_found) < 5:  # Only capture first 5
        for i in range(len(context)):
            if context[i] == Token.MEM and i + 8 < len(context):
                if len(mem_tokens_found) < 5:
                    addr_bytes = [context[i+1], context[i+2], context[i+3], context[i+4]]
                    val_bytes = [context[i+5], context[i+6], context[i+7], context[i+8]]
                    mem_tokens_found.append({
                        'addr': addr_bytes,
                        'val': val_bytes
                    })
    return original_extract(context)

runner._extract_mem_section = debug_extract

try:
    result = runner.run(bytecode, data, max_steps=15)
    print(f'   Result: {result}')

    if mem_tokens_found:
        print(f'\n3. First {len(mem_tokens_found)} MEM token(s):')
        for idx, mem in enumerate(mem_tokens_found):
            addr_hex = ''.join(f'{b:02x}' for b in mem['addr'])
            val_hex = ''.join(f'{b:02x}' for b in mem['val'])
            print(f'\n   MEM #{idx+1}:')
            print(f'     Addr: [{", ".join(f"0x{b:02x}" for b in mem["addr"])}] → 0x{addr_hex}')
            print(f'     Val:  [{", ".join(f"0x{b:02x}" for b in mem["val"])}] → 0x{val_hex}')

            if mem['addr'] == [0, 0, 0, 0]:
                print(f'     ⚠️  All-zero address!')
    else:
        print('\n⚠️  No MEM tokens found!')

except Exception as e:
    print(f'\n✗ Error: {e}')

print('=' * 70)
