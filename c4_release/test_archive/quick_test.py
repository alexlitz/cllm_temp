"""Quick test to check current opcode status."""
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

def test_opcode(name, bytecode):
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
    matches = 0
    mismatches = []
    for i in range(35):
        expected = draft_tokens[i]
        predicted = logits[0, ctx_len + i, :].argmax().item()
        if expected == predicted:
            matches += 1
        else:
            mismatches.append(i)

    pct = matches / 35 * 100
    if matches == 35:
        print(f'{name:10s}: 100.0% ✓')
    else:
        print(f'{name:10s}: {pct:5.1f}% ({matches}/35) - failing at positions {mismatches}')
    return matches == 35

print('Current opcode test results:')
print('=' * 70)
test_opcode('IMM', [Opcode.IMM | (5 << 8)])
test_opcode('NOP', [Opcode.NOP])
test_opcode('PSH', [Opcode.IMM | (42 << 8), Opcode.PSH])
test_opcode('ADD', [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.ADD])
test_opcode('JMP', [Opcode.JMP | (16 << 8)])
test_opcode('EXIT', [Opcode.EXIT])
