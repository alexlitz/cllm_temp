"""Simple test for JSR MEM generation."""

import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

code = '''
int helper() { return 42; }
int main() {
    helper();
    return 0;
}
'''

print('Compiling...')
bytecode, data = compile_c(code)
print(f'✓ Compiled: {len(bytecode)} bytes')

print('\nRunning VM...')
runner = AutoregressiveVMRunner()

# Patch runner to save context
saved_context = []
original_run = runner.run

def patched_run(*args, **kwargs):
    # Patch _extract_mem_section to save context
    original_extract = runner._extract_mem_section

    def save_context_extract(context):
        saved_context.append(list(context))
        return original_extract(context)

    runner._extract_mem_section = save_context_extract
    return original_run(*args, **kwargs)

runner.run = patched_run

# Run and capture context
try:
    result = runner.run(bytecode, data, max_steps=10)
    print(f'✓ Execution completed: {result}')

    # Check latest saved context for MEM tokens
    if not saved_context:
        print('\n⚠️  No context saved!')
        sys.exit(1)

    context = saved_context[-1]  # Use most recent context
    print(f'\nFinal context length: {len(context)}')

    # Find all MEM tokens
    mem_count = 0
    for i in range(len(context)):
        if context[i] == Token.MEM and i + 8 < len(context):
            mem_section = context[i:i+9]
            addr_bytes = [mem_section[j+1] for j in range(4)]
            val_bytes = [mem_section[j+5] for j in range(4)]

            addr_hex = [f'{b:02x}' for b in addr_bytes]
            val_hex = [f'{b:02x}' for b in val_bytes]

            addr_int = sum(b << (j*8) for j, b in enumerate(addr_bytes))
            val_int = sum(b << (j*8) for j, b in enumerate(val_bytes))

            mem_count += 1
            print(f'\nMEM #{mem_count} at context position {i}:')
            print(f'  Addr: {addr_hex} → 0x{addr_int:08x}')
            print(f'  Val:  {val_hex} → 0x{val_int:08x}')

            # Check for byte corruption (byte 3 copying byte 0)
            if addr_bytes[3] == addr_bytes[0] and addr_bytes[0] != 0:
                print(f'  ⚠️  ADDR BUG: byte 3 ({addr_bytes[3]}) copies byte 0!')
            else:
                print(f'  ✓ Addr looks OK')

            if val_bytes[3] == Token.STEP_END:
                print(f'  ⚠️  VAL BUG: byte 3 = STEP_END!')

            if mem_count >= 5:  # Only show first 5
                print('\n...(showing first 5 MEM tokens only)')
                break

    if mem_count == 0:
        print('\n⚠️  No MEM tokens found in context!')

except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()
