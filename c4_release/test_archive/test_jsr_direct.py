"""Direct test of JSR MEM generation at model level."""

import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD
from src.compiler import compile_c

# Compile test program
code = '''
int helper() { return 42; }
int main() {
    helper();  // JSR at PC=0
    return 0;
}
'''

print('Compiling...')
bytecode, data = compile_c(code)
print(f'✓ Compiled: {len(bytecode)} bytes')
print(f'First few instructions: {[hex(b) for b in bytecode[:20]]}')

# Create model
print('\nInitializing model...')
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)
print('✓ Model initialized')

# Build minimal context to trigger JSR
# After setup, PC should be at 0, which has JSR instruction
print('\nBuilding context for first JSR...')

# Build step-by-step context
context = []

# Initial setup tokens (CODE_START + bytecode + DATA_END)
context.append(Token.CODE_START)
for b in bytecode:
    context.append(b)
context.append(Token.DATA_END)

# Add initial state (one step)
# PC=0, AX=0, SP=0x00feffe0, BP=0x00feffe0, STACK0=0, MEM=(0,0), STEP_END
context.extend([
    Token.REG_PC, 0, 0, 0, 0,
    Token.REG_AX, 0, 0, 0, 0,
    Token.REG_SP, 0xe0, 0xff, 0xfe, 0x00,
    Token.REG_BP, 0xe0, 0xff, 0xfe, 0x00,
    Token.STACK0, 0, 0, 0, 0,
    Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0,
    Token.STEP_END
])

print(f'Initial context length: {len(context)}')
print(f'Bytecode at PC=0: 0x{bytecode[0]:02x} (should be JSR opcode)')

# Run one step to execute JSR
print('\nRunning step 1 (should execute JSR)...')
x = torch.tensor([context], dtype=torch.long)

# Generate next 35 tokens (one full step)
generated = []
for i in range(35):
    logits = model(x)
    next_token = torch.argmax(logits[0, -1]).item()
    generated.append(next_token)
    x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

    if next_token == Token.STEP_END:
        print(f'  Generated {i+1} tokens, hit STEP_END')
        break

print(f'\nGenerated step tokens: {len(generated)}')

# Find MEM token in generated step
mem_idx = None
for i, tok in enumerate(generated):
    if tok == Token.MEM:
        mem_idx = i
        break

if mem_idx is None:
    print('✗ No MEM token generated!')
    print(f'Generated tokens: {[f"{t:3d}" for t in generated[:20]]}...')
else:
    print(f'✓ Found MEM at position {mem_idx} in generated step')

    # Extract MEM section (9 tokens: MEM + 4 addr + 4 val)
    if mem_idx + 8 < len(generated):
        mem_section = generated[mem_idx:mem_idx+9]
        addr_bytes = [mem_section[j+1] for j in range(4)]
        val_bytes = [mem_section[j+5] for j in range(4)]

        addr_hex = [f'{b:02x}' for b in addr_bytes]
        val_hex = [f'{b:02x}' for b in val_bytes]

        addr_int = sum(b << (j*8) for j, b in enumerate(addr_bytes))
        val_int = sum(b << (j*8) for j, b in enumerate(val_bytes))

        print(f'\nMEM section:')
        print(f'  Addr: {addr_hex} → 0x{addr_int:08x}')
        print(f'  Val:  {val_hex} → 0x{val_int:08x}')

        # Expected values for JSR at PC=0
        # SP -= 8: 0x00feffe0 - 8 = 0x00feffd8
        # Return addr: PC + 5 = 5
        print(f'\nExpected:')
        print(f'  Addr: 0x00feffd8 (SP - 8)')
        print(f'  Val:  0x00000005 (PC + 5)')

        # Check for bugs
        if addr_bytes[3] == addr_bytes[0] and addr_bytes[0] != 0:
            print(f'\n✗ ADDR BUG: byte 3 ({addr_bytes[3]:02x}) copies byte 0!')
        elif addr_bytes[3] != 0:
            print(f'\n✗ ADDR BUG: byte 3 should be 0x00, got 0x{addr_bytes[3]:02x}')
        else:
            print(f'\n✓ Addr byte 3 correct (0x00)')

        if val_bytes != [5, 0, 0, 0]:
            print(f'✗ VAL BUG: expected [05, 00, 00, 00], got {val_hex}')
        else:
            print(f'✓ Val correct!')
    else:
        print(f'✗ Incomplete MEM section (only {len(generated) - mem_idx} tokens after MEM)')
