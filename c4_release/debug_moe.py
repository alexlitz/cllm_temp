"""Debug MoE routing correctness."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Build model with MoE
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])
print(f'Prefix context ({len(context)} tokens): {context}')
print()

# Generate with MoE routing as runner would
init_op = bytecode[0] & 0xFF
print(f'Setting initial opcode to {init_op} (IMM)')
model.set_active_opcode(init_op)

# Check what weights look like after activation
for i, block in enumerate(model.blocks):
    ffn = block.ffn
    if not isinstance(ffn.W_up, torch.nn.Parameter):
        print(f'  L{i}: W_up shape = {ffn.W_up.shape}')

# Generate tokens
gen_moe = []
ctx1 = list(context)
for j in range(40):
    t = model.generate_next(ctx1)
    ctx1.append(t)
    gen_moe.append(t)

# Now do same without MoE routing
model2 = AutoregressiveVM()
set_vm_weights(model2)
model2.compact(block_size=32)
model2.compact_moe()

ctx2 = list(context)
gen_nomoe = []
for j in range(40):
    t = model2.generate_next(ctx2)
    ctx2.append(t)
    gen_nomoe.append(t)

# Compare
print("\nToken-by-token comparison:")
for j in range(min(len(gen_moe), len(gen_nomoe))):
    match = '==' if gen_moe[j] == gen_nomoe[j] else 'DIFF'
    if gen_moe[j] != gen_nomoe[j]:
        print(f'Token {j:2d}: MoE={gen_moe[j]:4d}  NoRoute={gen_nomoe[j]:4d}  {match}')
    if gen_moe[j] == Token.HALT or gen_nomoe[j] == Token.HALT:
        break

if gen_moe == gen_nomoe:
    print('All tokens match!')
else:
    for i in range(min(len(gen_moe), len(gen_nomoe))):
        if gen_moe[i] != gen_nomoe[i]:
            print(f'First divergence at token {i}')
            print(f'  MoE:     {gen_moe[max(0,i-2):i+3]}')
            print(f'  NoRoute: {gen_nomoe[max(0,i-2):i+3]}')
            break
