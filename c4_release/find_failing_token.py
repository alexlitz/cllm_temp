"""Find which token is failing in the 97.1% tests."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

# Test IMM to find the failing token
bytecode = [Opcode.IMM | (5 << 8)]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)

# Check all 35 tokens
token_names = (
    ['REG_PC'] + [f'PC_b{i}' for i in range(4)] +
    ['REG_AX'] + [f'AX_b{i}' for i in range(4)] +
    ['REG_SP'] + [f'SP_b{i}' for i in range(4)] +
    ['REG_BP'] + [f'BP_b{i}' for i in range(4)] +
    ['STACK0'] + [f'ST_b{i}' for i in range(4)] +
    ['MEM'] + [f'MEM_a{i}' for i in range(4)] + [f'MEM_v{i}' for i in range(4)] +
    ['END']
)

mismatches = []
for i in range(35):
    expected = draft_tokens[i]
    predicted = logits[0, ctx_len + i, :].argmax().item()
    if expected != predicted:
        mismatches.append((i, token_names[i], expected, predicted))

print('IMM 5 - Token mismatches:')
for pos, name, exp, pred in mismatches:
    print(f'  Position {pos:2d} ({name:8s}): expected={exp:3d}, predicted={pred:3d}')

if not mismatches:
    print('  ✓ All tokens match!')
else:
    print(f'  Total mismatches: {len(mismatches)}/35')
