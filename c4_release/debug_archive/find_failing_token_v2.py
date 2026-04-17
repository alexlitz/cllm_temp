"""Find the specific failing token position."""
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

# Initialize model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

def find_mismatches(name, bytecode):
    """Find which tokens don't match."""
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)

    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    ctx_len = len(context)

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
        predicted = logits[0, ctx_len - 1 + i, :].argmax().item()
        if expected != predicted:
            mismatches.append((i, token_names[i], expected, predicted))

    if mismatches:
        print(f"\n{name}: {len(mismatches)}/35 mismatches")
        for pos, tok_name, exp, pred in mismatches:
            print(f"  Token {pos:2d} ({tok_name:8s}): expected={exp:3d}, predicted={pred:3d}")
    else:
        print(f"{name}: ✓ All 35 tokens match")

    return mismatches

print("Finding failing token positions:")
print("=" * 70)

# Test various opcodes
find_mismatches('IMM', [Opcode.IMM | (42 << 8)])
find_mismatches('NOP', [Opcode.NOP])
find_mismatches('EXIT', [Opcode.EXIT])
find_mismatches('ADD', [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD])
find_mismatches('LEA', [Opcode.LEA | (8 << 8)])
find_mismatches('JMP', [Opcode.JMP | (16 << 8)])
